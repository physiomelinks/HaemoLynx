"""Tests for graph module."""
import pytest
import numpy as np
import networkx as nx

from ImageLynx.graph import (
    build_graph_segment_skan_stitched_loops,
    reconnect_secondary_loop_edges,
    optimise_graph_topology_fixed,
    validate_skeleton_connection,
    safer_simple_remove_all_degree2_nodes,
    trivial_remove_all_degree2_nodes,
    create_trivial_merged_edge,
    smart_multigraph_degree2_removal,
    merge_edges_with_topology_improvement,
    prune_vascular_stubs,
    select_boundary_nodes_by_method,
)
from ImageLynx.graph._helpers import (
    get_line_points_3d,
    calculate_path_length,
    calculate_voxel_path_length,
    calculate_edge_length,
)

_STRAIGHT_3_VOXELS = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]


def _only_edge(G):
    """The single edge of G as (sorted end nodes, its data)."""
    (u, v, data), = G.edges(data=True)
    return tuple(sorted((u, v))), data


def _assert_merged_into_one_edge_0_2(G):
    """simple_graph's path 0-1-2 became one edge 0-2 along the three voxels, length 2."""
    ends, data = _only_edge(G)
    assert ends == (0, 2)
    assert data["voxels"] == _STRAIGHT_3_VOXELS
    assert data["length"] == 2.0
    return data


def test_get_line_points_3d():
    pts = get_line_points_3d(np.array([0, 0, 0]), np.array([3, 0, 0]))
    assert pts == [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]

    # A diagonal steps through 26-connected voxels: no step jumps a voxel on any axis.
    diag = get_line_points_3d(np.array([0, 0, 0]), np.array([3, 3, 3]))
    assert diag[0] == (0, 0, 0) and diag[-1] == (3, 3, 3)
    assert set(diag) == {(i, i, i) for i in range(4)}
    steps = np.abs(np.diff(np.array(diag), axis=0))
    assert steps.max() <= 1


def test_calculate_path_length():
    voxels = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    # Default isotropic
    assert calculate_path_length(voxels) == 2.0
    # Anisotropic (Z, Y, X)
    assert calculate_path_length(voxels, voxel_size=(2.0, 1.0, 1.0)) == 4.0


def test_calculate_voxel_path_length():
    voxels = [(0, 0, 0), (0, 1, 0), (0, 2, 0)]
    assert calculate_voxel_path_length(voxels) == 2.0
    assert calculate_voxel_path_length(voxels, voxel_size=(1.0, 2.0, 1.0)) == 4.0


def test_calculate_edge_length():
    # 1. Using actual voxels
    edge_data = {"voxels": [(0, 0, 0), (1, 0, 0), (2, 0, 0)]}
    assert calculate_edge_length(1, 2, edge_data, voxel_size=(2.0, 1.0, 1.0)) == 4.0

    # 2. Using pos attribute
    edge_data2 = {"pos": [(0, 0, 0), (2, 0, 0)]}
    assert calculate_edge_length(1, 2, edge_data2, voxel_size=(2.0, 1.0, 1.0)) == 4.0

    # 3. Using pre-calculated length
    edge_data3 = {"length": 5.5}
    assert calculate_edge_length(1, 2, edge_data3) == 5.5


def test_validate_skeleton_connection(tiny_skeleton):
    ok, path = validate_skeleton_connection(
        tiny_skeleton, np.array([2, 4, 4]), np.array([5, 4, 4])
    )
    assert (ok, path) == (True, [(2, 4, 4), (3, 4, 4), (4, 4, 4), (5, 4, 4)])

    # Two points well off the line: no skeleton near the straight path between them.
    assert validate_skeleton_connection(
        tiny_skeleton, np.array([2, 0, 0]), np.array([5, 0, 0])
    ) == (False, None)


def test_create_trivial_merged_edge():
    e1 = {"voxels": [(0, 0, 0), (1, 0, 0)], "weight": 1, "length": 1}
    e2 = {"voxels": [(1, 0, 0), (2, 0, 0)], "weight": 1, "length": 1}
    merged = create_trivial_merged_edge(e1, e2, np.array([1, 0, 0]))
    assert merged["weight"] == 2
    assert merged["voxels"] == _STRAIGHT_3_VOXELS   # the shared voxel is kept once
    assert merged["length"] == 2


def test_merge_edges_with_topology_improvement():
    v1 = [(0, 0, 0), (1, 0, 0)]
    v2 = [(1, 0, 0), (2, 0, 0)]
    skel = np.zeros((5, 5, 5))
    skel[1, 0, 0] = 1
    out = merge_edges_with_topology_improvement(
        v1, v2, np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([2, 0, 0]), skel
    )
    assert out == _STRAIGHT_3_VOXELS

    # Two curved edges are joined as they are (merge_curved_edges), not re-traced.
    c1 = [(0, 0, 0), (0, 1, 0), (1, 1, 0)]
    c2 = [(1, 1, 0), (2, 1, 0), (2, 0, 0)]
    curved = merge_edges_with_topology_improvement(
        c1, c2, np.array([0, 0, 0]), np.array([1, 1, 0]), np.array([2, 0, 0]), skel
    )
    assert curved == [(0, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0)]


def test_safer_simple_remove_all_degree2_nodes(simple_graph):
    G = safer_simple_remove_all_degree2_nodes(simple_graph.copy(), max_degree=5)
    _assert_merged_into_one_edge_0_2(G)


def test_trivial_remove_all_degree2_nodes(simple_graph):
    G = trivial_remove_all_degree2_nodes(simple_graph.copy(), max_degree=5)
    _assert_merged_into_one_edge_0_2(G)


def test_prune_vascular_stubs(simple_graph):
    # simple_graph is a plain nx.Graph: the branch the MultiGraph tests never reach.
    kept = prune_vascular_stubs(simple_graph.copy(), min_stub_length=0.5)
    assert sorted(kept.nodes()) == [0, 1, 2]
    assert sorted(kept.edges()) == [(0, 1), (1, 2)]

    # Both 1.0-long stubs are shorter than 1.5, so both go, leaving the middle node bare.
    pruned = prune_vascular_stubs(simple_graph.copy(), min_stub_length=1.5)
    assert sorted(pruned.nodes()) == [1]
    assert pruned.number_of_edges() == 0


def test_reconnect_secondary_loop_edges():
    # A skeleton ring between junctions u and v: a straight route A (21 voxels) and a
    # detour B (51 voxels, longer than min_length_voxels=30), plus a tail off each junction.
    # The graph holds only route A, so u and v are degree 2 there.
    # Arrays are (z, y, x); the ring lies in the plane z = 10.
    skeleton = np.zeros((21, 40, 41), dtype=bool)
    route_a = [(10, 10, x) for x in range(10, 31)]
    route_b = ([(10, y, 10) for y in range(10, 26)]
               + [(10, 25, x) for x in range(11, 30)]
               + [(10, y, 30) for y in range(25, 9, -1)])
    tail_u = [(10, 10, x) for x in range(2, 11)]
    tail_v = [(10, 10, x) for x in range(30, 39)]
    for voxel in route_a + route_b + tail_u + tail_v:
        skeleton[voxel] = True

    def physical(path):
        return [[float(c) for c in voxel] for voxel in path]

    G = nx.MultiGraph()
    for name, voxel in (("tail_u_end", tail_u[0]), ("u", route_a[0]),
                        ("v", route_a[-1]), ("tail_v_end", tail_v[-1])):
        G.add_node(name, pos=np.array(voxel, dtype=float))
    G.add_edge("tail_u_end", "u", voxels=physical(tail_u))
    G.add_edge("u", "v", voxels=physical(route_a))
    G.add_edge("v", "tail_v_end", voxels=physical(tail_v))

    G2 = reconnect_secondary_loop_edges(G, skeleton, debug=False)

    secondary = [(tuple(sorted((u, v))), d) for u, v, d in G2.edges(data=True)
                 if d.get("secondary")]
    assert len(secondary) == 1
    ends, data = secondary[0]
    assert ends == ("u", "v")
    voxels = [tuple(int(round(c)) for c in p) for p in data["voxels"]]
    assert {voxels[0], voxels[-1]} == {route_a[0], route_a[-1]}
    assert all(skeleton[v] for v in voxels)
    # It runs along the other route, out to its far side, not back along route A.
    assert set(voxels) <= set(route_b)
    assert (10, 25, 20) in voxels


def test_optimise_graph_topology_fixed():
    # Two segments a-b and c-d on one line, with terminals b and c 2 voxels apart.
    def two_segments():
        G = nx.MultiGraph()
        for name, p in {"a": (1, 5, 5), "b": (4, 5, 5), "c": (6, 5, 5), "d": (9, 5, 5)}.items():
            G.add_node(name, pos=np.array(p, dtype=float))
        G.add_edge("a", "b")
        G.add_edge("c", "d")
        return G

    through_gap = np.zeros((12, 12, 12), dtype=bool)
    through_gap[1:10, 5, 5] = True
    G, _ = optimise_graph_topology_fixed(two_segments(), [], set(),
                                         skeleton_data=through_gap, debug=False)
    assert sorted(tuple(sorted(e)) for e in G.edges()) == [("a", "b"), ("b", "c"), ("c", "d")]
    (join,) = G.get_edge_data("b", "c").values()
    assert join["validated"] is True
    assert join["voxels"] == [[4.0, 5.0, 5.0], [5.0, 5.0, 5.0], [6.0, 5.0, 5.0]]

    # The same pair, but the skeleton stops well short of the gap: not joined.
    short_of_gap = np.zeros((12, 12, 12), dtype=bool)
    short_of_gap[1:3, 5, 5] = True
    short_of_gap[8:10, 5, 5] = True
    G, _ = optimise_graph_topology_fixed(two_segments(), [], set(),
                                         skeleton_data=short_of_gap, debug=False)
    assert sorted(tuple(sorted(e)) for e in G.edges()) == [("a", "b"), ("c", "d")]


def test_smart_multigraph_degree2_removal(simple_graph):
    G = nx.MultiGraph(simple_graph)
    G2 = smart_multigraph_degree2_removal(G, skeleton_data=None, debug=False)
    data = _assert_merged_into_one_edge_0_2(G2)
    assert data["original_edges"] == 2

    # A degree-2 node whose two edges both go to the same neighbour is left alone:
    # merging them would collapse two parallel vessels into a self-loop.
    parallel = nx.MultiGraph()
    parallel.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    parallel.add_node(1, pos=np.array([2.0, 0.0, 0.0]))
    parallel.add_edge(0, 1, voxels=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    parallel.add_edge(0, 1, voxels=[(0, 0, 0), (1, 1, 0), (2, 0, 0)])
    after = smart_multigraph_degree2_removal(parallel, skeleton_data=None, debug=False)
    assert sorted(after.nodes()) == [0, 1]
    assert after.number_of_edges(0, 1) == 2
    assert nx.number_of_selfloops(after) == 0


def test_select_boundary_nodes_by_method_coordinates():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0]))
    G.add_node(1, pos=np.array([8.0, 8.0, 8.0]))
    G.add_node(2, pos=np.array([5.0, 5.0, 5.0]))
    G.add_edge(0, 2, length=1.0, weight=1.0)
    G.add_edge(1, 2, length=1.0, weight=1.0)

    nodes = select_boundary_nodes_by_method(
        G,
        (10, 10, 10),
        method="coordinates",
        node_role="input",
        coordinates=[(0.0, 0.0, 0.0)],
    )
    assert nodes == [0]


def test_select_boundary_nodes_by_method_volume_and_exclude():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0]))
    G.add_node(1, pos=np.array([8.0, 8.0, 8.0]))
    G.add_node(2, pos=np.array([5.0, 5.0, 5.0]))
    G.add_edge(0, 2, length=1.0, weight=1.0)
    G.add_edge(1, 2, length=1.0, weight=1.0)

    nodes = select_boundary_nodes_by_method(
        G,
        (10, 10, 10),
        method="volume",
        node_role="output",
        volume_boxes=[((0.0, 0.0, 0.0), (9.0, 9.0, 9.0))],
        exclude_nodes=[0],
    )
    assert nodes == [1]


def test_select_boundary_nodes_by_method_degree_1_from_starting():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([3.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([5.0, 0.0, 0.0]))
    G.add_node(3, pos=np.array([10.0, 0.0, 0.0]))
    G.add_edge(0, 2, length=1.0, weight=1.0)
    G.add_edge(1, 2, length=1.0, weight=1.0)
    G.add_edge(2, 3, length=1.0, weight=1.0)

    nodes = select_boundary_nodes_by_method(
        G,
        (20, 20, 20),
        method="degree_1_from_starting",
        node_role="output",
        starting_nodes_for_distance=[0],
        distance_from_starting_node=5.0,
        exclude_nodes=[0],
    )
    assert nodes == [3]

def test_validate_voxel_path_continuity():
    from ImageLynx.graph._helpers import validate_voxel_path_continuity
    
    voxels = [(0, 0, 0), (1, 0, 0), (3, 0, 0)] # Gap of 2
    gap = validate_voxel_path_continuity(voxels)
    assert np.isclose(gap, 2.0)
    
    voxels_continuous = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert np.isclose(validate_voxel_path_continuity(voxels_continuous), 1.0)


def test_resolve_core_dead_ends_safe_zone_protection():
    from ImageLynx.graph.prune import resolve_core_dead_ends
    
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 50, 50)) # Boundary safe zone (Z=0)
    G.add_node(1, pos=(50, 50, 50)) # Core (Z=50)
    G.add_edge(0, 1) # Note: Node 0 is Degree 1 (Boundary). Node 1 is Degree 1 (Core).
    
    stats = resolve_core_dead_ends(G.copy(), (100, 100, 100), (1.0, 1.0, 1.0), "eradicate", 5.0)
    assert stats["edges_removed"] == 1
    
    # What if a node is entirely in the safe zone?
    G2 = nx.MultiGraph()
    G2.add_node(0, pos=(0, 50, 50)) # Safe
    G2.add_node(1, pos=(1, 50, 50)) # Safe
    G2.add_edge(0, 1)
    stats2 = resolve_core_dead_ends(G2.copy(), (100, 100, 100), (1.0, 1.0, 1.0), "eradicate", 5.0)
    assert stats2.get("edges_removed", 0) == 0 # Neither node is core, so neither is queued.

def test_resolve_core_dead_ends_eradicate_mode():
    from ImageLynx.graph.prune import resolve_core_dead_ends
    G = nx.MultiGraph()
    G.add_node(0, pos=(50, 50, 50)) # Core dead-end
    G.add_node(1, pos=(51, 50, 50)) # Core
    G.add_node(2, pos=(52, 50, 50)) # Core (will be bifurcation)
    G.add_node(3, pos=(53, 50, 50)) # Core
    G.add_node(4, pos=(54, 50, 50)) # Core
    
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    stats = resolve_core_dead_ends(G, (100, 100, 100), (1.0, 1.0, 1.0), "eradicate", 5.0)
    assert stats["edges_removed"] == 4
    assert stats["edges_removed_pct"] == 100.0

def test_resolve_core_dead_ends_stitch_mode():
    from ImageLynx.graph.prune import resolve_core_dead_ends
    G = nx.MultiGraph()
    # Dead end at Z=50
    G.add_node(0, pos=(50, 50, 50)) # Core
    G.add_node(1, pos=(0, 50, 50)) # Boundary safe
    G.add_edge(0, 1) # 0 is deg 1 core.
    
    # Valid target at Z=55
    G.add_node(2, pos=(55, 50, 50)) # Core
    G.add_node(3, pos=(99, 50, 50)) # Boundary safe
    G.add_edge(2, 3) # 2 is deg 1 core.
    
    stats = resolve_core_dead_ends(G, (100, 100, 100), (1.0, 1.0, 1.0), "stitch", 5.0, 10.0, 4)
    assert stats["edges_added"] == 1
    assert G.has_edge(0, 2) or G.has_edge(2, 0)
    
def test_resolve_core_dead_ends_stitch_degree_limit():
    from ImageLynx.graph.prune import resolve_core_dead_ends
    G = nx.MultiGraph()
    G.add_node(0, pos=(50, 50, 50)) # Core dead-end
    G.add_node(1, pos=(0, 50, 50)) # Boundary safe
    G.add_edge(0, 1)
    
    # Node 2 is close (dist 2) but has Degree=4
    G.add_node(2, pos=(52, 50, 50))
    for i in range(10, 14):
        G.add_node(i, pos=(99, 50, i)) # safe
        G.add_edge(2, i) # 2 is now Degree 4
        
    # Node 3 is further (dist 5) but has Degree=2
    G.add_node(3, pos=(55, 50, 50))
    G.add_node(15, pos=(99, 60, 50)) # safe
    G.add_node(16, pos=(99, 65, 50)) # safe
    G.add_edge(3, 15)
    G.add_edge(3, 16) # 3 is now Degree 2
    
    stats = resolve_core_dead_ends(G, (100, 100, 100), (1.0, 1.0, 1.0), "stitch", 5.0, 10.0, 4)
    assert G.has_edge(0, 3) or G.has_edge(3, 0)
    assert not G.has_edge(0, 2)
    assert stats["edges_added"] == 1

def test_resolve_core_dead_ends_stitch_fallback():
    from ImageLynx.graph.prune import resolve_core_dead_ends
    G = nx.MultiGraph()
    G.add_node(0, pos=(50, 50, 50))
    G.add_node(1, pos=(40, 50, 50))
    G.add_edge(0, 1) # 0 is deg 1
    
    # No other nodes in the graph!
    stats = resolve_core_dead_ends(G, (100, 100, 100), (1.0, 1.0, 1.0), "stitch", 5.0, 10.0, 4)
    # Should fallback and eradicate
    assert stats["fallback_eradicated"] == 1
    assert stats["edges_removed"] == 1
    assert len(G.edges()) == 0

def test_chaikin_and_resample():
    from ImageLynx.graph._helpers import _chaikin_once, _resample_polyline
    points = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [10.0, 0.0, 0.0]])
    
    smoothed = _chaikin_once(points)
    assert smoothed[0].tolist() == [0.0, 0.0, 0.0]
    assert smoothed[-1].tolist() == [10.0, 0.0, 0.0]
    assert len(smoothed) > 3
    
    resampled = _resample_polyline(smoothed, 5)
    assert resampled.shape == (5, 3)
    assert resampled[0].tolist() == [0.0, 0.0, 0.0]
    assert resampled[-1].tolist() == [10.0, 0.0, 0.0]


def test_direct_skan_loop_extraction():
    pytest.importorskip("skan")
    from skan import csr
    
    # Create a 3D block with a known loop
    data = np.zeros((10, 10, 10), dtype=bool)
    data[1, 1, 1:4] = True
    data[1, 3, 1:4] = True
    data[1, 1:4, 1] = True
    data[1, 1:4, 3] = True
    
    sk = csr.Skeleton(data)
    
    G, loops, loop_edges = build_graph_segment_skan_stitched_loops(
        sk, data, debug=True
    )
    
    assert len(loops) >= 1
    assert len(loops[0]) == 8 # The loop has 8 voxels


def _loop_skeleton():
    from skan import csr

    data = np.zeros((10, 10, 10), dtype=bool)
    data[1, 1, 1:4] = True
    data[1, 3, 1:4] = True
    data[1, 1:4, 1] = True
    data[1, 1:4, 3] = True
    return csr.Skeleton(data), data


def test_loop_extraction_without_igraph_raises(monkeypatch):
    """It used to return no loops, so the graph lost its loop edges and nothing said so."""
    pytest.importorskip("skan")
    import builtins

    sk, data = _loop_skeleton()
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "igraph":
            raise ImportError("Mocked missing igraph")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="python-igraph"):
        build_graph_segment_skan_stitched_loops(sk, data, debug=True)


def test_a_loop_extraction_error_is_not_swallowed(monkeypatch):
    pytest.importorskip("skan")
    igraph = pytest.importorskip("igraph")

    def broken(*args, **kwargs):
        raise RuntimeError("biconnected components failed")

    monkeypatch.setattr(igraph.Graph, "biconnected_components", broken)
    sk, data = _loop_skeleton()
    with pytest.raises(RuntimeError, match="biconnected components failed"):
        build_graph_segment_skan_stitched_loops(sk, data, debug=True)

