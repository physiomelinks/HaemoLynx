"""Tests for graph module."""
import pytest
import numpy as np
import networkx as nx

from haemolynx.graph import (
    build_graph_segment_skan_stitched_loops,
    reconnect_secondary_loop_edges,
    optimise_graph_topology_fixed,
    reconnect_orphan_and_dangling_nodes,
    validate_skeleton_connection,
    safer_simple_remove_all_degree2_nodes,
    trivial_remove_all_degree2_nodes,
    create_trivial_merged_edge,
    duplicate_parallel_edges,
    duplicate_vessel_routes,
    remove_duplicate_parallel_edges,
    smart_multigraph_degree2_removal,
    prune_vascular_stubs,
    assign_branch_orders,
    select_boundary_nodes_by_method,
    select_boundary_terminal_nodes,
    select_terminal_nodes_from_large_vessel_masks,
    diagnose_degree2_nodes,
    format_degree2_diagnostics_report,
)
from haemolynx.graph.optimise import _reconnection_is_direction_safe
from haemolynx.graph._helpers import (
    get_line_points_3d,
    calculate_path_length,
    calculate_edge_length,
    add_edge_safe,
    get_all_edge_data,
    are_paths_similar,
    path_separation,
    paths_separated_by_background,
    should_add_merged_edge,
)
from haemolynx.graph.assemble import mask_lumen_test


def test_get_line_points_3d():
    pts = get_line_points_3d(np.array([0, 0, 0]), np.array([3, 0, 0]))
    assert len(pts) >= 2
    assert pts[0] == (0, 0, 0)
    assert pts[-1] == (3, 0, 0)


def test_calculate_path_length():
    voxels = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert calculate_path_length(voxels) == 2.0


def test_validate_skeleton_connection(tiny_skeleton):
    ok, path = validate_skeleton_connection(
        tiny_skeleton, np.array([2, 4, 4]), np.array([5, 4, 4])
    )
    assert isinstance(ok, bool)
    assert path is None or isinstance(path, list)


def test_create_trivial_merged_edge():
    e1 = {"voxels": [(0, 0, 0), (1, 0, 0)], "length": 1}
    e2 = {"voxels": [(1, 0, 0), (2, 0, 0)], "length": 1}
    merged = create_trivial_merged_edge(e1, e2, np.array([1, 0, 0]))
    assert "weight" not in merged
    assert merged["length"] == 2
    assert len(merged["voxels"]) >= 3


def _degree2_count(G):
    return sum(1 for n in G.nodes() if G.degree[n] == 2)


def _total_edge_length(G):
    edges = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    return sum(item[-1].get("length", 0.0) for item in edges)


def test_safer_simple_remove_all_degree2_nodes(simple_graph):
    before = simple_graph.copy()
    G = safer_simple_remove_all_degree2_nodes(simple_graph.copy(), max_degree=5)
    # The point of the pass: no removable degree-2 node may survive it.
    assert _degree2_count(G) == 0
    assert G.number_of_nodes() < before.number_of_nodes()
    # A rejected merge must not silently drop vessel length.
    assert _total_edge_length(G) == pytest.approx(_total_edge_length(before))


def test_trivial_remove_all_degree2_nodes(simple_graph):
    before = simple_graph.copy()
    G = trivial_remove_all_degree2_nodes(simple_graph.copy(), max_degree=5)
    assert _degree2_count(G) == 0
    assert G.number_of_nodes() < before.number_of_nodes()
    assert _total_edge_length(G) == pytest.approx(_total_edge_length(before))


def test_prune_vascular_stubs(simple_graph):
    G = prune_vascular_stubs(simple_graph.copy(), min_stub_length=0.5)
    assert G.number_of_nodes() >= 1


def test_assign_branch_orders(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    G.add_node(2, pos=np.array([10.0, 0.0, 0.0]))
    G.add_edge(1, 2, weight=1.0, length=5.0, voxels=[(5, 0, 0), (10, 0, 0)])
    res = assign_branch_orders(G, [0])
    assert res["edges_assigned"] >= 1


def test_reconnect_secondary_loop_edges(tiny_skeleton):
    pytest.importorskip("skan")
    from skan import csr
    import networkx as nx
    sk = csr.Skeleton(tiny_skeleton)
    G = build_graph_segment_skan_stitched_loops(sk, tiny_skeleton)
    G = nx.MultiGraph(G)
    G2 = reconnect_secondary_loop_edges(G, tiny_skeleton, debug=False)
    assert G2.number_of_nodes() == G.number_of_nodes()


def _degree2_chain_graph():
    """A 5-node straight chain: the middle 1-2 and 2-3 edges are both

    deg==2 -> deg==2, so reconnect_secondary_loop_edges actually attempts
    them (an all-terminal fixture like ``tiny_skeleton``'s never does).
    """
    G = nx.MultiGraph()
    for i in range(5):
        G.add_node(i, pos=np.array([float(i), 4.0, 4.0]))
    for i in range(4):
        G.add_edge(
            i, i + 1, length=1.0, voxels=[(i, 4, 4), (i + 1, 4, 4)]
        )
    return G


def test_reconnect_reports_a_failure_summary_even_without_debug_logging(
    monkeypatch, caplog
):
    """Regression: subvolume/pathfinding/metric exceptions used to be logged

    only ``if debug`` -- off by default (tied to verbose_logging) -- so a run
    where every candidate pair hit the same exception looked identical to a
    run that cleanly found nothing to reconnect. A count must reach the log
    regardless of ``debug``.
    """
    from haemolynx.graph import reconnect as reconnect_module

    def _always_raises(*_args, **_kwargs):
        raise RuntimeError("synthetic EDT failure")

    monkeypatch.setattr(reconnect_module, "distance_transform_edt", _always_raises)

    skeleton = np.zeros((8, 8, 8), dtype=bool)
    skeleton[0:5, 4, 4] = True

    with caplog.at_level("WARNING"):
        result = reconnect_module.reconnect_secondary_loop_edges(
            _degree2_chain_graph(), skeleton, debug=False
        )

    assert result.number_of_nodes() == 5
    assert any(
        "subvolume creation failed" in record.message for record in caplog.records
    )


def test_optimise_graph_topology_fixed(tiny_skeleton):
    pytest.importorskip("skan")
    from skan import csr
    import networkx as nx
    sk = csr.Skeleton(tiny_skeleton)
    G = build_graph_segment_skan_stitched_loops(sk, tiny_skeleton)
    G2 = optimise_graph_topology_fixed(G, skeleton_data=tiny_skeleton, debug=False)
    assert isinstance(G2, nx.Graph)


def _wheel_target(n_existing: int = 5, spoke_length: float = 10.0) -> nx.MultiGraph:
    """A node at the origin with *n_existing* spokes evenly spread around a
    circle in the y-z plane -- the same "wheel" arrangement
    test_direction_aware_collapse.py uses, just for one node's own incident
    edges rather than a cluster of nodes being merged."""
    import math

    G = nx.MultiGraph()
    G.add_node("tgt", pos=np.array([0.0, 0.0, 0.0]))
    for i in range(n_existing):
        angle = 2 * math.pi * i / n_existing
        pos = np.array([0.0, spoke_length * math.cos(angle), spoke_length * math.sin(angle)])
        name = f"n{i}"
        G.add_node(name, pos=pos)
        G.add_edge("tgt", name, key=0, length=spoke_length, voxels=[[0.0, 0.0, 0.0], pos.tolist()])
    return G


def test_reconnection_is_direction_safe_below_min_degree_is_always_safe():
    """Fewer than min_degree_for_dispersion_check spokes (existing + the
    candidate): an ordinary bifurcation cannot look wheel-shaped."""
    G = _wheel_target(n_existing=4)
    candidate = np.array([0.0, 10.0, 0.0])
    assert _reconnection_is_direction_safe(
        G, "tgt", np.array([0.0, 0.0, 0.0]), candidate,
        min_degree_for_dispersion_check=6, max_radial_dispersion=0.5,
        tangent_length_um=10.0,
    )


def test_reconnection_is_direction_safe_rejects_completing_a_wheel():
    """5 existing evenly-spaced spokes plus a 6th continuing the same even
    spread is exactly the cartwheel shape this check exists to catch."""
    import math

    G = _wheel_target(n_existing=5)
    candidate = np.array(
        [0.0, 10.0 * math.cos(2 * math.pi * 2.5 / 5), 10.0 * math.sin(2 * math.pi * 2.5 / 5)]
    )
    assert not _reconnection_is_direction_safe(
        G, "tgt", np.array([0.0, 0.0, 0.0]), candidate,
        min_degree_for_dispersion_check=6, max_radial_dispersion=0.5,
        tangent_length_um=10.0,
    )


def test_reconnection_is_direction_safe_allows_a_coherent_direction():
    """5 existing spokes all pointing roughly the same way, plus a 6th
    agreeing with them, stays coherent even past min_degree."""
    G = nx.MultiGraph()
    G.add_node("tgt", pos=np.array([0.0, 0.0, 0.0]))
    for i in range(5):
        jitter = (i - 2) * 2.0
        pos = np.array([20.0, jitter, 0.0])
        G.add_node(f"n{i}", pos=pos)
        G.add_edge("tgt", f"n{i}", key=0, length=20.0, voxels=[[0.0, 0.0, 0.0], pos.tolist()])
    candidate = np.array([20.0, 1.0, 0.0])
    assert _reconnection_is_direction_safe(
        G, "tgt", np.array([0.0, 0.0, 0.0]), candidate,
        min_degree_for_dispersion_check=6, max_radial_dispersion=0.5,
        tangent_length_um=10.0,
    )


def test_reconnect_orphan_and_dangling_nodes_direction_aware_skips_a_wheel_completing_reconnection():
    """End-to-end: an orphan sitting exactly where it would complete tgt's
    wheel is reconnected when direction_aware is off (today's default
    behaviour, unchanged) and skipped when it is on."""
    import math

    def _graph_with_orphan():
        # Close to tgt (radius 2, well inside the spoke tips' own radius 10),
        # so tgt -- not one of the spoke-tip nodes, also nearby -- is
        # unambiguously the nearest reconnection target. Only the *angle*
        # (matching the wheel-completing direction) needs to be exact.
        G = _wheel_target(n_existing=5)
        angle = 2 * math.pi * 2.5 / 5
        orphan_pos = np.array([0.0, 2.0 * math.cos(angle), 2.0 * math.sin(angle)])
        G.add_node("orphan", pos=orphan_pos)
        return G

    without = reconnect_orphan_and_dangling_nodes(
        _graph_with_orphan(), reconnect_threshold=5.0, validate_reconnections=False,
    )
    assert without.has_edge("tgt", "orphan")

    with_guard = reconnect_orphan_and_dangling_nodes(
        _graph_with_orphan(), reconnect_threshold=5.0, validate_reconnections=False,
        direction_aware=True, max_radial_dispersion=0.5, min_degree_for_dispersion_check=6,
        tangent_length_um=10.0,
    )
    assert not with_guard.has_edge("tgt", "orphan")
    # The orphan is genuinely left disconnected, not connected some other way.
    assert with_guard.degree["orphan"] == 0


def test_smart_multigraph_degree2_removal(simple_graph):
    G = nx.MultiGraph(simple_graph)
    before_nodes = G.number_of_nodes()
    before_length = _total_edge_length(G)
    G2 = smart_multigraph_degree2_removal(G, skeleton_data=None, debug=False)
    assert isinstance(G2, nx.MultiGraph)
    # Previously asserted only the return type, so an empty body would have passed.
    assert _degree2_count(G2) == 0
    assert G2.number_of_nodes() < before_nodes
    assert _total_edge_length(G2) == pytest.approx(before_length)


def _two_spurs_bridged_at_their_tips():
    """A vessel A-J-X-K-B (with a side branch X-C) and two spurs, J-N (bent)
    and K-M (straight), whose tips a gap bridge N-M joins: the shape the
    build step's terminal reconnection leaves. N and M are degree 2. The
    skeleton has no voxels across the gap, so the only skeleton route from N
    to M runs back down J-N, along the vessel and up K-M."""
    z = 4
    skeleton = np.zeros((8, 20, 26), dtype=bool)
    paths = {
        ("A", "J"): [(z, 8, x) for x in range(0, 7)],
        ("J", "X"): [(z, 8, x) for x in range(6, 13)],
        ("X", "K"): [(z, 8, x) for x in range(12, 19)],
        ("K", "B"): [(z, 8, x) for x in range(18, 25)],
        ("X", "C"): [(z, y, 12) for y in range(8, 1, -1)],
        ("J", "N"): [(z, 8, 6), (z, 9, 7), (z, 10, 8), (z, 11, 9), (z, 12, 9),
                     (z, 13, 8), (z, 14, 7), (z, 15, 6), (z, 16, 6)],
        ("K", "M"): [(z, y, 18) for y in range(8, 17)],
    }
    G = nx.MultiGraph(voxel_size=(1.0, 1.0, 1.0))
    for name in ("A", "J", "X", "K", "B", "C", "N", "M"):
        end = next(p[0] if a == name else p[-1] for (a, b), p in paths.items() if name in (a, b))
        G.add_node(name, pos=np.array(end, dtype=float))
    for (a, b), path in paths.items():
        skeleton[tuple(np.array(path).T)] = True
        voxels = [tuple(float(c) for c in p) for p in path]
        G.add_edge(a, b, voxels=voxels, length=calculate_path_length(voxels))
    bridge = [tuple(G.nodes["N"]["pos"]), tuple(G.nodes["M"]["pos"])]
    G.add_edge("N", "M", voxels=bridge, length=calculate_path_length(bridge), reconnected=True)
    return G, skeleton


def test_degree2_merge_joins_a_spur_and_a_gap_bridge_without_retracing():
    """Regression: merging a degree-2 node re-traced a straight leg through the
    skeleton by A*. A gap bridge has no skeleton across its gap, so the trace
    from spur tip N to M went back down the spur it had just come up and along
    the vessel -- an edge that ran out to N and back over the same voxels,
    duplicating the vessel's own, and here measured 61% too long. On an
    E14.5 stack, 209 of 3772 edges ran out and back like this.

    A merged edge is its two legs, each run once."""
    G, skeleton = _two_spurs_bridged_at_their_tips()
    lengths = {(a, b): d["length"] for a, b, d in G.edges(data=True)}
    loop_length = lengths[("J", "N")] + lengths[("N", "M")] + lengths[("K", "M")]

    merged = smart_multigraph_degree2_removal(G.copy(), skeleton_data=skeleton, debug=False)

    assert not merged.has_node("N") and not merged.has_node("M")
    for u, v, data in merged.edges(data=True):
        points = [tuple(np.round(p, 6)) for p in data["voxels"]]
        assert len(points) == len(set(points)), f"edge {u}-{v} passes a point twice"
    (loop,) = [d for u, v, d in merged.edges(data=True) if {u, v} == {"J", "K"}]
    assert loop["length"] == pytest.approx(loop_length)
    assert _total_edge_length(merged) == pytest.approx(_total_edge_length(G))


def test_degree2_diagnostics(simple_graph):
    report = diagnose_degree2_nodes(simple_graph, max_degree=4)
    assert "total_degree2" in report
    assert "reason_counts" in report
    text = format_degree2_diagnostics_report(report)
    assert "Degree-2 diagnostics" in text


def _line(start, end, n_points):
    """`n_points` evenly spaced physical points from *start* to *end*."""
    return [tuple(p) for p in np.linspace(start, end, n_points)]


def _junction_pair(a=(0.0, 0.0, 0.0), b=(0.0, 0.0, 20.0), stubs=1):
    """Junctions "a" and "b" with *stubs* terminal branches each, so neither is degree-2 itself."""
    G = nx.MultiGraph()
    for name, pos in (("a", a), ("b", b)):
        G.add_node(name, pos=np.asarray(pos, dtype=float))
    for i in range(stubs):
        offset = np.array([10.0 * i, 0.0, 10.0])
        for junction, stub_pos in (("a", np.subtract(a, offset)), ("b", np.add(b, offset))):
            stub = f"stub_{junction}{i}"
            G.add_node(stub, pos=stub_pos)
            _add_path_edge(G, junction, stub, _line(G.nodes[junction]["pos"], stub_pos, 11))
    return G


def _add_path_edge(G, u, v, voxels):
    length = calculate_path_length(voxels)
    G.add_edge(u, v, voxels=voxels, length=length)
    return length


def _add_route_through(G, mid, mid_pos, n_points=11):
    """A two-edge route a -> *mid* -> b, *mid* being the degree-2 node to remove."""
    G.add_node(mid, pos=np.asarray(mid_pos, dtype=float))
    return _add_path_edge(G, "a", mid, _line(G.nodes["a"]["pos"], mid_pos, n_points)) + _add_path_edge(
        G, mid, "b", _line(mid_pos, G.nodes["b"]["pos"], n_points)
    )


def _straight_and_bowed_routes():
    G = _junction_pair()
    straight = _add_path_edge(G, "a", "b", _line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21))
    bow = _add_route_through(G, "bow", (0.0, 10.0, 10.0))
    return G, [straight, bow]


def _two_bows_of_different_length():
    G = _junction_pair()
    near = _add_route_through(G, "near_bow", (0.0, 6.0, 10.0))
    far = _add_route_through(G, "far_bow", (0.0, -12.0, 10.0))
    return G, [near, far]


def _short_vessel_and_folded_hairpin():
    """A 3 um vessel a-b, and a spur out of "a" whose gap bridge folds back to "b"."""
    G = _junction_pair(b=(0.0, 0.0, 3.0))
    vessel = _add_path_edge(G, "a", "b", _line((0.0, 0.0, 0.0), (0.0, 0.0, 3.0), 4))
    hairpin = _add_route_through(G, "spur_tip", (0.0, 15.0, 0.0), n_points=16)
    return G, [vessel, hairpin]


@pytest.mark.parametrize(
    "build",
    [_straight_and_bowed_routes, _two_bows_of_different_length, _short_vessel_and_folded_hairpin],
)
def test_degree2_removal_keeps_distinct_parallel_vessels(build):
    """Two different routes between the same junctions are two vessels, not one.

    The duplicate check used to compare only the paths' end points, which any
    two parallel edges share, so merging the route through the degree-2 node
    either replaced the other vessel (curved beats straight, or >5% shorter) or
    was refused and left the degree-2 node behind.
    """
    G, vessel_lengths = build()
    before_length = _total_edge_length(G)

    G2 = smart_multigraph_degree2_removal(G, skeleton_data=None, debug=False)

    assert _degree2_count(G2) == 0
    assert _total_edge_length(G2) == pytest.approx(before_length)
    # Both vessels are still there, as parallel a-b edges at their own lengths.
    parallel_lengths = sorted(d["length"] for d in G2["a"]["b"].values())
    assert parallel_lengths == pytest.approx(sorted(vessel_lengths))


def test_degree2_removal_still_drops_a_route_that_retraces_an_existing_edge():
    """A merged path lying along an existing a-b edge is that vessel twice: one copy stays."""
    # Two stubs each, so "a" and "b" are still junctions once the copy is gone.
    G = _junction_pair(stubs=2)
    straight = _add_path_edge(G, "a", "b", _line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21))
    # The same vessel traced again, as a voxel staircase never more than 1 um off it.
    staircase = [(0.0, float(i % 2), float(i)) for i in range(21)]
    G.add_node("mid", pos=np.asarray(staircase[10], dtype=float))
    _add_path_edge(G, "a", "mid", staircase[:11])
    _add_path_edge(G, "mid", "b", staircase[10:])
    before_length = _total_edge_length(G)

    G2 = smart_multigraph_degree2_removal(G, skeleton_data=None, max_degree=8, debug=False)

    assert not G2.has_node("mid")
    assert G2.number_of_edges("a", "b") == 1
    # The straight copy gave way to the staircase one (curved is preferred).
    assert _total_edge_length(G2) == pytest.approx(before_length - straight)


def test_path_separation_is_the_furthest_either_path_strays():
    straight = _line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21)
    assert path_separation(straight, [(0.0, 5.0, p[2]) for p in straight]) == pytest.approx(5.0)
    # A two-point chord is sampled along its length, not only at its ends.
    bow = [(0.0, 0.0, 0.0), (0.0, 10.0, 10.0), (0.0, 0.0, 20.0)]
    chord = [(0.0, 0.0, 0.0), (0.0, 0.0, 20.0)]
    assert path_separation(chord, bow) == pytest.approx(10.0, abs=0.25)
    assert path_separation(bow, chord) == path_separation(chord, bow)


def test_path_separation_can_leave_out_the_samples_near_a_point():
    straight = [(0.0, 0.0, 0.0), (0.0, 0.0, 20.0)]
    # The same vessel, except that it starts 4 um to the side and cuts back in.
    side_start = [(0.0, 4.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 20.0)]
    assert path_separation(straight, side_start) == pytest.approx(4.0)
    assert path_separation(straight, side_start, ignore_within=((0.0, 0.0, 0.0), 5.0)) == pytest.approx(0.0)
    # Outside the ball the far side still counts, measured to the whole other path.
    bow = [(0.0, 0.0, 0.0), (0.0, 10.0, 10.0), (0.0, 0.0, 20.0)]
    assert path_separation(straight, bow, ignore_within=((0.0, 0.0, 0.0), 5.0)) == pytest.approx(10.0, abs=0.25)
    # Nothing left to measure: nothing tells the paths apart.
    assert path_separation(straight, bow, ignore_within=((0.0, 0.0, 10.0), 50.0)) == 0.0


def test_paths_with_shared_end_points_are_similar_only_if_they_coincide():
    straight = _line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21)
    staircase = [(0.0, float(i % 2), float(i)) for i in range(21)]
    bow = _line((0.0, 0.0, 0.0), (0.0, 10.0, 10.0), 11) + _line((0.0, 10.0, 10.0), (0.0, 0.0, 20.0), 11)[1:]
    assert are_paths_similar(straight, staircase)
    assert are_paths_similar(straight, staircase[::-1])
    assert not are_paths_similar(straight, bow)
    # Coinciding along the way is not enough when the ends differ.
    assert not are_paths_similar(straight, straight[:12])


def test_should_add_merged_edge_replaces_only_a_coinciding_edge():
    G = _junction_pair()
    straight = _line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21)
    key = G.add_edge("a", "b", voxels=straight, length=20.0)
    staircase = [(0.0, float(i % 2), float(i)) for i in range(21)]
    bow = _line((0.0, 0.0, 0.0), (0.0, 10.0, 10.0), 11) + _line((0.0, 10.0, 10.0), (0.0, 0.0, 20.0), 11)[1:]

    assert should_add_merged_edge(
        G, "a", "b", staircase, {"length": calculate_path_length(staircase)}
    ) == (True, key)
    assert should_add_merged_edge(
        G, "a", "b", bow, {"length": calculate_path_length(bow)}
    ) == (True, None)


def _lumen(shape, *boxes):
    """`inside_lumen` for a mask of 1 um voxels, True inside each box of index slices."""
    mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        mask[box] = True
    return mask_lumen_test(mask, (1.0, 1.0, 1.0))


def test_paths_separated_by_background_reads_the_mask_between_them():
    """Two strands 7 um apart are one vessel or two depending on what lies
    between them -- Lee thinning leaves strands that far apart through one
    wide vessel, so distance cannot tell."""
    a = _line((5.0, 10.0, 5.0), (5.0, 10.0, 35.0), 31)
    b = _line((5.0, 17.0, 5.0), (5.0, 17.0, 35.0), 31)
    shape = (11, 30, 41)
    one_lumen = _lumen(shape, np.s_[2:9, 7:21, :])
    two_vessels = _lumen(shape, np.s_[2:9, 8:13, :], np.s_[2:9, 15:20, :])  # 2 um of tissue between
    only_a = _lumen(shape, np.s_[2:9, 8:13, :])

    assert not paths_separated_by_background(a, b, one_lumen)
    assert paths_separated_by_background(a, b, two_vessels)
    assert paths_separated_by_background(b, a, two_vessels)
    # A path running outside the mask is no evidence of a second vessel.
    assert not paths_separated_by_background(a, b, only_a)
    # Nor is anything between paths that coincide.
    assert not paths_separated_by_background(a, a, two_vessels)


def test_paths_separated_by_background_needs_most_of_the_way_separated():
    """A vessel with a short hole in its lumen (a cell, a dark speck) is
    still one vessel; tissue between the strands most of the way is two."""
    a = _line((5.0, 10.0, 5.0), (5.0, 10.0, 35.0), 31)
    b = _line((5.0, 17.0, 5.0), (5.0, 17.0, 35.0), 31)
    shape = (11, 30, 41)
    holed = np.zeros(shape, dtype=bool)
    holed[2:9, 7:21, :] = True
    holed[:, 12:16, 17:23] = False  # 6 um of the 30 um have tissue between
    assert not paths_separated_by_background(a, b, mask_lumen_test(holed, (1.0, 1.0, 1.0)))
    holed[:, 12:16, 8:32] = False  # 24 of the 30
    assert paths_separated_by_background(a, b, mask_lumen_test(holed, (1.0, 1.0, 1.0)))


def _square_detour(width=7.0):
    """From "a" (0, 0, 0) out *width* um, along, and back to "b" (0, 0, 20)."""
    return (
        _line((0.0, 0.0, 0.0), (0.0, width, 4.0), 9)
        + _line((0.0, width, 4.0), (0.0, width, 16.0), 13)[1:]
        + _line((0.0, width, 16.0), (0.0, 0.0, 20.0), 9)[1:]
    )


#: Masks for _square_detour beside the straight a-b vessel: one lumen round
#: both, or two vessels 4 um apart that meet at the two junctions.
_DETOUR_SHAPE = (2, 10, 22)
_ONE_LUMEN = (np.s_[0:1, 0:9, 0:21],)
_TWO_VESSELS = (
    np.s_[0:1, 0:2, 0:21],
    np.s_[0:1, 6:9, 0:21],
    np.s_[0:1, 0:9, 0:3],
    np.s_[0:1, 0:9, 18:21],
)


def test_are_paths_similar_counts_two_strands_through_one_lumen_as_one_vessel():
    straight = _line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21)
    detour = _square_detour()
    # 7 um apart: two vessels, by distance alone.
    assert not are_paths_similar(straight, detour)
    assert are_paths_similar(straight, detour, inside_lumen=_lumen(_DETOUR_SHAPE, *_ONE_LUMEN))
    assert not are_paths_similar(straight, detour, inside_lumen=_lumen(_DETOUR_SHAPE, *_TWO_VESSELS))


@pytest.mark.parametrize("route_is", ["the detour", "straight"])
def test_degree2_removal_keeps_one_vessel_of_two_strands_through_one_lumen(route_is):
    """Regression test: strands 5-10 um apart through one lumen are one
    vessel. Judged by distance alone (> 3 um apart) the merged route was
    added beside the existing edge: pairs of parallel edges in the E14.5 MCA
    graph went from 23 to 157, nearly all inside one segmented vessel.
    Whichever copy is kept, the
    other goes -- including a refused route, which used to stay behind as
    its degree-2 node and two edges."""
    G = _junction_pair(stubs=2)
    detour, straight = _square_detour(), _line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21)
    edge, route = (straight, detour) if route_is == "the detour" else (detour, straight)
    _add_path_edge(G, "a", "b", edge)
    G.add_node("mid", pos=np.asarray(route[len(route) // 2], dtype=float))
    _add_path_edge(G, "a", "mid", route[: len(route) // 2 + 1])
    _add_path_edge(G, "mid", "b", route[len(route) // 2 :])
    nodes_before = G.number_of_nodes()

    G2 = smart_multigraph_degree2_removal(
        G.copy(), max_degree=8, inside_lumen=_lumen(_DETOUR_SHAPE, *_ONE_LUMEN)
    )

    assert not G2.has_node("mid")
    assert G2.number_of_nodes() == nodes_before - 1
    # One a-b vessel, and it is the detour either way: curved is preferred to straight.
    assert [d["length"] for d in G2["a"]["b"].values()] == pytest.approx([calculate_path_length(detour)])


def test_degree2_removal_keeps_both_strands_when_the_mask_separates_them():
    G = _junction_pair(stubs=2)
    straight = _add_path_edge(G, "a", "b", _line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21))
    detour = _square_detour()
    G.add_node("mid", pos=np.asarray(detour[len(detour) // 2], dtype=float))
    _add_path_edge(G, "a", "mid", detour[: len(detour) // 2 + 1])
    _add_path_edge(G, "mid", "b", detour[len(detour) // 2 :])

    G2 = smart_multigraph_degree2_removal(
        G, max_degree=8, inside_lumen=_lumen(_DETOUR_SHAPE, *_TWO_VESSELS)
    )

    assert not G2.has_node("mid")
    assert sorted(d["length"] for d in G2["a"]["b"].values()) == pytest.approx(
        sorted([straight, calculate_path_length(detour)])
    )


def _ab_edges(*paths, stubs=2):
    """Junctions "a" and "b" (see _junction_pair) joined directly by each of *paths*, keys 0, 1, ..."""
    G = _junction_pair(stubs=stubs)
    for path in paths:
        _add_path_edge(G, "a", "b", path)
    return G


@pytest.mark.parametrize(
    "lumen, duplicate_keys",
    [
        # Without a mask only the staircase's straight twin, 1 um off it, repeats.
        (None, {0}),
        # One lumen: all three are one vessel; the detour (curved, shortest) stays.
        (_ONE_LUMEN, {0, 1}),
        # Tissue between the detour and the other two: two vessels, one of them twice.
        (_TWO_VESSELS, {0}),
    ],
)
def test_duplicate_parallel_edges_names_every_copy_but_one_of_each_vessel(lumen, duplicate_keys):
    straight = _line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21)
    staircase = [(0.0, float(i % 2), float(i)) for i in range(21)]
    G = _ab_edges(straight, staircase, _square_detour())
    inside_lumen = None if lumen is None else _lumen(_DETOUR_SHAPE, *lumen)

    duplicates = duplicate_parallel_edges(G, inside_lumen)

    assert {key for _, _, key in duplicates} == duplicate_keys
    assert remove_duplicate_parallel_edges(G, inside_lumen) == len(duplicate_keys)
    assert set(G["a"]["b"]) == {0, 1, 2} - duplicate_keys
    assert duplicate_parallel_edges(G, inside_lumen) == []


def test_degree2_removal_removes_duplicate_parallel_edges_it_did_not_make():
    """Regression test: skeletonisation leaves loops round one lumen as two
    edges between the same junctions, with no degree-2 node between them for
    a merge to compare -- so no step removed them, and at E14.5 MCA 20 such
    pairs came straight through from the first graph build."""
    detour = _square_detour()
    G = _ab_edges(_line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21), detour)

    kept_apart = smart_multigraph_degree2_removal(
        G.copy(), max_degree=8, inside_lumen=_lumen(_DETOUR_SHAPE, *_TWO_VESSELS)
    )
    one_vessel = smart_multigraph_degree2_removal(
        G.copy(), max_degree=8, inside_lumen=_lumen(_DETOUR_SHAPE, *_ONE_LUMEN)
    )

    assert kept_apart.number_of_edges("a", "b") == 2
    assert [d["length"] for d in one_vessel["a"]["b"].values()] == pytest.approx(
        [calculate_path_length(detour)]
    )
    assert set(one_vessel.nodes) == set(G.nodes)


@pytest.mark.parametrize("lumen", ["one", "two"])
def test_degree2_removal_resolves_a_duplicate_route_even_beside_a_busy_junction(lumen):
    """Regression test: merging leaves a degree-2 node alone beside a junction
    of max_degree or more, and with it any route through it repeating the
    edge its neighbours already share -- at E14.5 MCA three were left that
    way, one vessel drawn twice. A route the mask separates from that edge
    is a vessel of its own and stays as it was."""
    G = _ab_edges(_line((0.0, 0.0, 0.0), (0.0, 0.0, 20.0), 21), stubs=4)
    detour = _square_detour()
    G.add_node("mid", pos=np.asarray(detour[len(detour) // 2], dtype=float))
    _add_path_edge(G, "a", "mid", detour[: len(detour) // 2 + 1])
    _add_path_edge(G, "mid", "b", detour[len(detour) // 2 :])
    inside_lumen = _lumen(_DETOUR_SHAPE, *(_ONE_LUMEN if lumen == "one" else _TWO_VESSELS))
    assert G.degree["a"] >= 4
    assert duplicate_vessel_routes(G, inside_lumen) == (["mid"] if lumen == "one" else [])

    G2 = smart_multigraph_degree2_removal(G.copy(), max_degree=4, inside_lumen=inside_lumen)

    assert duplicate_vessel_routes(G2, inside_lumen) == []
    if lumen == "one":
        assert not G2.has_node("mid")
        assert G2.number_of_edges("a", "b") == 1
    else:
        assert G2.has_node("mid") and G2.number_of_edges("a", "b") == 1


def test_build_graph_requires_skan(tiny_skeleton):
    pytest.importorskip("skan")
    from skan import csr
    sk = csr.Skeleton(tiny_skeleton)
    G = build_graph_segment_skan_stitched_loops(sk, tiny_skeleton, debug=False)
    assert isinstance(G, nx.MultiGraph)
    assert G.number_of_nodes() == 2 and G.number_of_edges() == 1


def _short_segments_skeleton(count: int) -> np.ndarray:
    """*count* separate 4-voxel segments along z, each with its two ends 3 um
    apart and every segment 10 voxels from the next."""
    skeleton = np.zeros((8, 8, 10 * count), dtype=bool)
    for i in range(count):
        skeleton[2:6, 4, 10 * i + 4] = True
    return skeleton


@pytest.mark.parametrize("segments", [1, 6])
def test_a_segment_is_not_bridged_to_itself(segments):
    """Two ends of one short segment lie within the reconnect threshold of each
    other, so they are a candidate pair -- but joining them would draw a
    second, parallel copy of the segment. Up to 10 terminals the candidates
    come from a plain double loop, above that from a k-d tree; neither may
    offer such a pair. The fold-back rule is off so that nothing else would
    refuse it."""
    pytest.importorskip("skan")
    from skan import csr

    skeleton = _short_segments_skeleton(segments)
    G = build_graph_segment_skan_stitched_loops(
        csr.Skeleton(skeleton), skeleton, reconnect_threshold=3.5, max_bridge_turn_deg=None
    )

    assert sum(1 for n in G if G.degree[n] == 1) == 2 * segments
    assert G.number_of_edges() == segments
    assert not any(data.get("reconnected") for *_, data in G.edges(data=True))


def test_building_the_graph_does_not_search_the_skeleton_for_voxel_cycles(monkeypatch):
    """Regression: every build turned each skeleton voxel into a voxel-graph
    node and ran ``nx.cycle_basis`` over it -- ~22 s of the E14.5 build -- and
    the offsets included (0, 0, 0), so every voxel was a "loop". What it found
    only ever excluded terminal pairs an edge already joined. A real vessel
    loop in the skeleton still comes out as a cycle in the graph."""
    pytest.importorskip("skan")
    from skan import csr

    searched = []
    cycle_basis = nx.cycle_basis
    # Recorded, not raised: the old block caught any exception and carried on.
    monkeypatch.setattr(nx, "cycle_basis", lambda *a, **k: searched.append(a) or cycle_basis(*a, **k))
    # A square ring with a branch leaving it at three places, so the ring is
    # three edges between three junctions rather than one self-loop.
    skeleton = np.zeros((5, 24, 24), dtype=bool)
    skeleton[2, 4, 4:20] = skeleton[2, 19, 4:20] = True
    skeleton[2, 4:20, 4] = skeleton[2, 4:20, 19] = True
    skeleton[2, 19:23, 12] = skeleton[2, 1:4, 12] = skeleton[2, 12, 1:4] = True

    G = build_graph_segment_skan_stitched_loops(csr.Skeleton(skeleton), skeleton)

    monkeypatch.undo()
    assert searched == []
    assert len(nx.cycle_basis(nx.Graph(G))) == 1


def _network_short_of_the_image_border(scale: float = 1.0) -> nx.MultiGraph:
    """A Y-shaped network whose terminals stop well inside a 48-voxel image.

    ``scale`` is the voxel size in microns, which is what node ``pos`` is
    measured in; the graph is the same either way.
    """
    positions = {
        0: (24.0, 8.0, 24.0),   # the one terminal at the low-y end
        1: (24.0, 20.0, 24.0),
        2: (24.0, 32.0, 12.0),
        3: (24.0, 32.0, 36.0),
        4: (24.0, 41.0, 8.0),   # the two terminals at the high-y end
        5: (24.0, 41.0, 40.0),
    }
    G = nx.MultiGraph()
    for node_id, position in positions.items():
        G.add_node(node_id, pos=np.asarray(position, dtype=float) * scale)
    for u, v in ((0, 1), (1, 2), (1, 3), (2, 4), (3, 5)):
        G.add_edge(u, v, length=1.0)
    return G


def test_edge_percent_bands_span_the_network_not_the_image():
    """Terminals inside the image, not on its border, are still selected.

    The bands used to be measured across the image, so a network that stops
    short of the border -- which pruning and stub removal make the normal case
    -- fell outside both of them and no boundary node was found at all.
    """
    G = _network_short_of_the_image_border()

    starting, outputs = select_boundary_terminal_nodes(
        G, (48, 48, 48), edge_percent=10.0, end_percent=10.0, axis=1
    )

    assert starting == [0]
    assert sorted(outputs) == [4, 5]


def test_edge_percent_bands_do_not_depend_on_the_voxel_size():
    """`pos` is in microns and `image_shape` counts voxels.

    Comparing the two only agreed at 1 micron voxels: at 0.25 micron the whole
    network sat below the old outlet band and no outlet existed.
    """
    G = _network_short_of_the_image_border(scale=0.25)

    starting, outputs = select_boundary_terminal_nodes(
        G, (48, 48, 48), edge_percent=10.0, end_percent=10.0, axis=1
    )

    assert starting == [0]
    assert sorted(outputs) == [4, 5]


def test_edge_percent_bands_are_disjoint_even_when_they_overlap():
    """Wide bands still name each terminal once, as an inlet or an outlet."""
    G = _network_short_of_the_image_border()

    starting, outputs = select_boundary_terminal_nodes(
        G, (48, 48, 48), edge_percent=90.0, end_percent=90.0, axis=1
    )

    assert set(starting).isdisjoint(outputs)
    assert starting and outputs


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
        node_role="inlet",
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
        node_role="outlet",
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
        method="degree_1_from_inlet",
        node_role="outlet",
        inlet_nodes_for_distance=[0],
        distance_from_inlet_node=5.0,
        exclude_nodes=[0],
    )
    assert nodes == [3]


def test_select_terminal_nodes_from_large_vessel_masks():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0]))
    G.add_node(1, pos=np.array([8.0, 8.0, 8.0]))
    G.add_node(2, pos=np.array([5.0, 5.0, 5.0]))
    G.add_edge(0, 2, length=1.0, weight=1.0)
    G.add_edge(1, 2, length=1.0, weight=1.0)

    arteriole_mask = np.zeros((10, 10, 10), dtype=bool)
    venule_mask = np.zeros((10, 10, 10), dtype=bool)
    arteriole_mask[1, 1, 1] = True
    venule_mask[8, 8, 8] = True

    start_nodes, out_nodes = select_terminal_nodes_from_large_vessel_masks(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    assert start_nodes == [0]
    assert out_nodes == [1]


def test_select_terminal_nodes_from_large_vessel_masks_excludes_overlap():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([4.0, 4.0, 4.0]))
    G.add_node(1, pos=np.array([5.0, 5.0, 5.0]))
    G.add_node(2, pos=np.array([4.5, 4.5, 4.5]))
    G.add_edge(0, 2, length=1.0, weight=1.0)
    G.add_edge(1, 2, length=1.0, weight=1.0)

    arteriole_mask = np.zeros((10, 10, 10), dtype=bool)
    venule_mask = np.zeros((10, 10, 10), dtype=bool)
    arteriole_mask[4, 4, 4] = True
    venule_mask[4, 4, 4] = True
    venule_mask[5, 5, 5] = True

    start_nodes, out_nodes = select_terminal_nodes_from_large_vessel_masks(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        allow_overlap=False,
    )
    assert start_nodes == [0]
    assert out_nodes == [1]



def _loop_in_the_z_x_plane(spacing=(2.0, 0.5, 0.5)):
    """A rectangular skeleton loop: the edge between u (x=10) and v (x=70)
    runs along its bottom (z=2); the other way round runs along its top, four
    2 um slices -- 8 um -- above. Tails make u and v degree 2."""
    skeleton = np.zeros((12, 5, 80), dtype=bool)
    skeleton[2, 2, 0:80] = True
    skeleton[6, 2, 10:71] = True
    skeleton[2:7, 2, 10] = skeleton[2:7, 2, 70] = True
    s = np.asarray(spacing)

    def physical(z, x):
        return [float(z * s[0]), float(2 * s[1]), float(x * s[2])]

    G = nx.MultiGraph()
    for node, x in {0: 0, 1: 10, 2: 70, 3: 79}.items():
        G.add_node(node, pos=np.array(physical(2, x)))
    G.add_edge(0, 1, voxels=[physical(2, x) for x in range(0, 11)])
    G.add_edge(1, 2, voxels=[physical(2, x) for x in range(10, 71)])
    G.add_edge(2, 3, voxels=[physical(2, x) for x in range(70, 80)])
    return G, skeleton


def test_a_loop_8_um_away_in_z_is_not_too_similar_to_add():
    """Regression (audit): the "too similar" test was a Hausdorff distance in
    voxel indices, so the other way round a loop four 2 um slices (8 um) away
    was 4 voxels from the edge -- under the 8 the edge must differ by -- and
    refused, where the same 8 um in-plane (16 voxels of 0.5 um) was added."""
    G, skeleton = _loop_in_the_z_x_plane()

    result = reconnect_secondary_loop_edges(
        G, skeleton, voxel_size=(2.0, 0.5, 0.5), debug=False, max_workers=1, routing_processes=0
    )

    secondary = [d for _u, _v, d in result.edges(data=True) if d.get("secondary")]
    assert len(secondary) == 1
    top = np.asarray(secondary[0]["voxels"])
    assert top[:, 0].max() == pytest.approx(12.0)  # z = 6 slices x 2 um
    # Its length is the physical way round: 8 up, 30 across, 8 down.
    assert secondary[0]["length"] == pytest.approx(46.0, abs=1.5)


def test_a_traced_skeleton_path_is_the_shortest_in_microns():
    """Regression (audit): A* counted voxel steps, so between two points it
    took the route with fewest voxels -- here one slice up through z and
    straight across (8 steps, 6.9 um) -- over the in-plane way round an
    obstacle (12 steps, 6.4 um, and shorter)."""
    from haemolynx.graph._helpers import trace_skeleton_path

    spacing = (2.0, 0.5, 0.5)
    skeleton = np.zeros((3, 9, 12), dtype=bool)
    skeleton[0, 4, 0:3] = skeleton[0, 4, 9:12] = True  # the two ends, in slice 0
    skeleton[0, 1, 3:9] = True  # round the obstacle, in-plane ...
    skeleton[0, 2, 2] = skeleton[0, 3, 2] = skeleton[0, 2, 9] = skeleton[0, 3, 9] = True
    skeleton[1, 4, 3:9] = True  # ... or one slice up and straight across

    path = trace_skeleton_path(skeleton, (0.0, 2.0, 0.0), (0.0, 2.0, 5.5), voxel_size=spacing)

    assert path is not None
    assert {round(p[0], 6) for p in path} == {0.0}  # never leaves slice 0


def test_a_connection_is_near_the_skeleton_within_the_same_physical_tolerance_every_way():
    """Regression (audit): "near" was the 3x3x3 voxel neighbourhood -- 2 um
    in z but 0.5 um in-plane on 2 x 0.5 x 0.5 um voxels -- so a line one
    whole z slice off the skeleton still validated."""
    spacing = (2.0, 0.5, 0.5)
    skeleton = np.zeros((6, 10, 30), dtype=bool)
    skeleton[3, 5, :] = True  # the skeleton, one slice above the line below

    one_slice_off, _ = validate_skeleton_connection(
        skeleton, np.array([4.0, 2.5, 1.0]), np.array([4.0, 2.5, 13.0]), voxel_size=spacing
    )
    on_it, _ = validate_skeleton_connection(
        skeleton, np.array([6.0, 2.5, 1.0]), np.array([6.0, 2.5, 13.0]), voxel_size=spacing
    )
    one_voxel_off_in_plane, _ = validate_skeleton_connection(
        skeleton, np.array([6.0, 3.0, 1.0]), np.array([6.0, 3.0, 13.0]), voxel_size=spacing
    )

    assert not one_slice_off
    assert on_it and one_voxel_off_in_plane


def _spur_beside_a_vessel_end_skeleton() -> np.ndarray:
    """One z slice: vessel A along x with an 8-voxel spur off its junction J
    (+y, tip N); vessel B ending at M two voxels short of A, beside J; and
    vessel C broken by a 4-voxel gap whose two ends face each other.

    N and M are 10 voxels apart, inside the reconnect threshold the tests
    use: a straight bridge between them runs from the spur's tip back to
    beside the junction the spur left.
    """
    skeleton = np.zeros((9, 64, 64), dtype=bool)
    skeleton[4, 30, 2:62] = True  # A; junction J at (4, 30, 30)
    skeleton[4, 31:39, 30] = True  # the spur off J, tip N at (4, 38, 30)
    skeleton[4, 6:29, 31] = True  # B; end M at (4, 28, 31)
    skeleton[4, 50, 2:20] = True  # C ...
    skeleton[4, 50, 24:44] = True  # ... and its other half, across the gap
    return skeleton


_SPUR_TIP = (4.0, 38.0, 30.0)
_VESSEL_B_END = (4.0, 28.0, 31.0)
_VESSEL_C_GAP = frozenset({(4.0, 50.0, 19.0), (4.0, 50.0, 24.0)})


def _bridged_end_pairs(G) -> set:
    return {
        frozenset(tuple(float(c) for c in G.nodes[n]["pos"]) for n in (u, v))
        for u, v, data in G.edges(data=True)
        if data.get("reconnected")
    }


def _sharpest_turn_deg(voxels) -> float:
    points = np.asarray(voxels, dtype=float)
    steps = np.diff(points, axis=0)
    steps = steps[np.linalg.norm(steps, axis=1) > 1e-9]
    if len(steps) < 2:
        return 0.0
    steps /= np.linalg.norm(steps, axis=1)[:, None]
    cosines = np.einsum("ij,ij->i", steps[:-1], steps[1:])
    return float(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).max())


def test_a_gap_bridge_does_not_fold_back_against_its_terminals_vessel():
    """Regression: terminals within the reconnect threshold were bridged
    nearest first with no regard to direction, so the spur's tip N was
    bridged to M -- straight back past the junction the spur came from --
    while the gap C's two facing ends span is still to be closed."""
    pytest.importorskip("skan")
    from skan import csr

    skeleton = _spur_beside_a_vessel_end_skeleton()

    G = build_graph_segment_skan_stitched_loops(
        csr.Skeleton(skeleton), skeleton, reconnect_threshold=12.0
    )
    unguarded = build_graph_segment_skan_stitched_loops(
        csr.Skeleton(skeleton), skeleton, reconnect_threshold=12.0, max_bridge_turn_deg=None
    )

    folding_back = frozenset({_SPUR_TIP, _VESSEL_B_END})
    assert folding_back in _bridged_end_pairs(unguarded)
    assert _bridged_end_pairs(G) == {_VESSEL_C_GAP}


def test_a_built_network_has_no_hairpin_where_a_spur_tip_was_bridged_beside_its_junction():
    """Regression: degree-2 merging turned junction -> spur tip -> bridge ->
    vessel end beside the junction into one edge running 8 um out and
    straight back (a 174 degree turn). Vessel B still joins A, at the
    junction rather than through the spur."""
    from haemolynx.graph import build_graph_from_skeleton

    G = build_graph_from_skeleton(_spur_beside_a_vessel_end_skeleton(), graph_reconnect_threshold=12.0)

    assert max(_sharpest_turn_deg(data["voxels"]) for *_, data in G.edges(data=True)) < 100.0
    by_position = {tuple(float(c) for c in G.nodes[n]["pos"]): n for n in G.nodes}
    assert nx.has_path(G, by_position[(4.0, 6.0, 31.0)], by_position[(4.0, 30.0, 2.0)])


def _straight_edge(G, a, b, **attrs):
    start, end = np.asarray(G.nodes[a]["pos"], float), np.asarray(G.nodes[b]["pos"], float)
    n = int(np.ceil(np.linalg.norm(end - start)))
    voxels = [(start + (end - start) * t).tolist() for t in np.linspace(0.0, 1.0, n + 1)]
    G.add_edge(a, b, length=float(np.linalg.norm(end - start)), voxels=voxels, **attrs)


def _graph_from(positions, edges) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.graph["voxel_size"] = (1.0, 1.0, 1.0)
    for name, pos in positions.items():
        G.add_node(name, pos=np.asarray(pos, dtype=float))
    for a, b in edges:
        _straight_edge(G, a, b)
    return G


def _two_terminal_pairs() -> nx.MultiGraph:
    """Spur J->N heading +y, with the end M of a vessel coming down z 1.4 um
    behind and beside N; and two vessel ends P, Q 1 um apart, facing."""
    return _graph_from(
        {
            "J": (0, 0, 0), "N": (0, 6, 0),
            "D": (0, 5, 10), "M": (0, 5, 1),
            "C": (0, 20, 0), "P": (0, 30, 0),
            "E": (0, 40, 0), "Q": (0, 31, 0),
        },
        [("J", "N"), ("D", "M"), ("C", "P"), ("E", "Q")],
    )


def test_optimise_graph_topology_does_not_join_terminals_folding_back():
    """The same rule as the initial gap bridges: N -> M turns 135 degrees from
    the way the spur was heading, so only P and Q are joined."""
    def joined(G):
        return {frozenset((u, v)) for u, v, d in G.edges(data=True) if d.get("reconnected")}

    old = optimise_graph_topology_fixed(
        _two_terminal_pairs(), reconnect_threshold=3.0,
        validate_reconnections=False, max_bridge_turn_deg=None,
    )
    new = optimise_graph_topology_fixed(
        _two_terminal_pairs(), reconnect_threshold=3.0, validate_reconnections=False,
    )

    assert joined(old) == {frozenset("NM"), frozenset("PQ")}
    assert joined(new) == {frozenset("PQ")}


def test_a_short_junction_arm_is_not_bridged_however_well_it_lines_up():
    """skan leaves 1-3 um arms round a junction's voxel cluster. They are not
    vessel ends, so the end R of a vessel stopping 1 um short of the arm S
    off junction H is not bridged to it, though the two point at each other."""
    def arm_facing_a_vessel_end():
        return _graph_from(
            {
                "F": (0, 50, 0), "R": (0, 60, 0),
                "H0": (0, 62, -10), "H": (0, 62, 0), "H1": (0, 62, 10), "S": (0, 61, 0),
            },
            [("F", "R"), ("H0", "H"), ("H", "H1"), ("H", "S")],
        )

    old = optimise_graph_topology_fixed(
        arm_facing_a_vessel_end(), reconnect_threshold=3.0,
        validate_reconnections=False, max_bridge_turn_deg=None,
    )
    new = optimise_graph_topology_fixed(
        arm_facing_a_vessel_end(), reconnect_threshold=3.0, validate_reconnections=False,
    )

    assert old.has_edge("R", "S")
    assert not new.has_edge("R", "S")


def _dangling_ends_beside_vessels() -> nx.MultiGraph:
    """Spur J->N heading +y with a vessel running along z through T, 2.8 um
    behind and beside N; and spur K->S heading +y with a vessel through U
    2 um straight ahead of S."""
    return _graph_from(
        {
            "J": (0, 0, 0), "N": (0, 6, 0),
            "T0": (0, 4, -8), "T": (0, 4, 2), "T1": (0, 4, 12),
            "K": (20, 0, 0), "S": (20, 6, 0),
            "U0": (20, 8, -10), "U": (20, 8, 0), "U1": (20, 8, 10),
        },
        [("J", "N"), ("T0", "T"), ("T", "T1"), ("K", "S"), ("U0", "U"), ("U", "U1")],
    )


def test_orphan_reconnection_does_not_fold_a_dangling_end_back():
    """A dangling end is joined to a vessel it runs into (S -> U), not to
    one lying behind it (N -> T)."""
    old = reconnect_orphan_and_dangling_nodes(
        _dangling_ends_beside_vessels(), reconnect_threshold=3.0,
        validate_reconnections=False, max_bridge_turn_deg=None,
    )
    new = reconnect_orphan_and_dangling_nodes(
        _dangling_ends_beside_vessels(), reconnect_threshold=3.0, validate_reconnections=False,
    )

    assert old.has_edge("N", "T") and old.has_edge("S", "U")
    assert not new.has_edge("N", "T") and new.has_edge("S", "U")
    assert new.degree["N"] == 1


# --- reconnects held to the segmentation ---------------------------------------


def _mask_support(mask):
    from haemolynx.preprocessing import MaskSupport

    return MaskSupport(mask, (1.0, 1.0, 1.0))


def _broken_vessel(background_voxels: int):
    """A centreline broken at x=12..14 (terminals at x=11 and x=15), in a
    tube of mask with *background_voxels* of background in that break."""
    skeleton = np.zeros((5, 5, 30), dtype=bool)
    skeleton[2, 2, 2:12] = True
    skeleton[2, 2, 15:28] = True
    mask = np.zeros_like(skeleton)
    mask[1:4, 1:4, :] = True
    first = 12 + (3 - background_voxels) // 2
    mask[:, :, first:first + background_voxels] = False
    return skeleton, mask


@pytest.mark.parametrize("background_voxels, bridged", [(1, True), (3, False)])
def test_a_gap_bridge_is_drawn_only_through_the_mask(background_voxels, bridged):
    """Regression: every pair of facing terminals within the threshold was
    bridged with a straight line, though three microns of background lay
    between them."""
    pytest.importorskip("skan")
    from skan import csr

    skeleton, mask = _broken_vessel(background_voxels)
    ends = frozenset({(2.0, 2.0, 11.0), (2.0, 2.0, 15.0)})

    plain = build_graph_segment_skan_stitched_loops(
        csr.Skeleton(skeleton), skeleton, reconnect_threshold=5.0
    )
    gated = build_graph_segment_skan_stitched_loops(
        csr.Skeleton(skeleton), skeleton, reconnect_threshold=5.0,
        mask_support=_mask_support(mask),
    )

    assert ends in _bridged_end_pairs(plain)
    assert (ends in _bridged_end_pairs(gated)) is bridged
    for *_, data in plain.edges(data=True):
        assert "bridge_kind" not in data
    for *_, data in gated.edges(data=True):
        if data.get("reconnected"):
            assert data["bridge_kind"] == "gap"
            assert data["bridge_background_um"] == pytest.approx(1.0, abs=0.05)
            assert data["length"] == pytest.approx(4.0)


def test_a_bridge_round_a_bend_follows_the_mask_not_the_chord():
    """The straight line between two ends of an L-shaped vessel cuts the
    corner through background; the route through the mask goes round it."""
    from haemolynx.graph.reconnect import MaskBridges, route_through_mask

    mask = np.zeros((3, 15, 15), dtype=bool)
    mask[0:3, 1:4, 1:14] = True  # along x at y=2
    mask[0:3, 1:14, 11:14] = True  # along y at x=12
    start, end = np.array([1.0, 2.0, 5.0]), np.array([1.0, 9.0, 12.0])
    support = _mask_support(mask)

    routed = route_through_mask(mask, start, end, (1.0, 1.0, 1.0))
    found = MaskBridges(nx.MultiGraph(), support, "gap").bridge(start, end)

    assert not support.accepts(np.vstack([start, end]))
    assert np.array_equal(routed[0], start) and np.array_equal(routed[-1], end)
    assert support.accepts(routed)
    assert found is not None and len(found[0]) > 2
    assert found[1].longest_background_um < 1.0


def _a_dangling_end_facing_a_vessel(joined_by_mask: bool):
    """Spur K -> S heading +y, ending 4 um short of a vessel through U; the
    mask either runs on from the spur into the vessel or leaves three
    microns of background between them."""
    G = _graph_from(
        {
            "K": (5, 2, 10), "S": (5, 8, 10),
            "U0": (5, 12, 0), "U": (5, 12, 10), "U1": (5, 12, 20),
        },
        [("K", "S"), ("U0", "U"), ("U", "U1")],
    )
    mask = np.zeros((11, 16, 22), dtype=bool)
    mask[4:7, 11:14, :] = True  # the vessel, y 11..13
    mask[4:7, 1:(11 if joined_by_mask else 8), 9:12] = True  # the spur
    return G, mask


@pytest.mark.parametrize("joined_by_mask", [True, False])
def test_an_orphan_reconnect_needs_the_mask_between_its_ends(joined_by_mask):
    G, mask = _a_dangling_end_facing_a_vessel(joined_by_mask)
    settings = dict(reconnect_threshold=5.0, validate_reconnections=False)

    plain = reconnect_orphan_and_dangling_nodes(G.copy(), **settings)
    gated = reconnect_orphan_and_dangling_nodes(G.copy(), mask_support=_mask_support(mask), **settings)

    assert plain.has_edge("S", "U")
    assert gated.has_edge("S", "U") is joined_by_mask
    if joined_by_mask:
        data = gated.edges["S", "U", 0]
        assert data["bridge_kind"] == "orphan" and data["bridge_background_um"] == 0.0


@pytest.mark.parametrize("segments", [1, 6])
def test_optimise_does_not_bridge_a_segment_to_itself(segments):
    """The optimise pass's own candidate pairs, as for the initial gap bridges:
    a segment whose two ends lie 1 um apart is not joined end to end, through
    the double loop (2 terminals) or the k-d tree (12)."""
    positions, edges = {}, []
    for i in range(segments):
        positions[f"a{i}"], positions[f"b{i}"] = (0, 0, 20 * i), (1, 0, 20 * i)
        edges.append((f"a{i}", f"b{i}"))

    G = optimise_graph_topology_fixed(
        _graph_from(positions, edges), reconnect_threshold=3.0,
        validate_reconnections=False, max_bridge_turn_deg=None,
    )

    assert G.number_of_edges() == segments
    assert not any(data.get("reconnected") for *_, data in G.edges(data=True))


def test_an_optimise_reconnect_is_tagged_with_what_it_crossed():
    plain = optimise_graph_topology_fixed(
        _two_terminal_pairs(), reconnect_threshold=3.0, validate_reconnections=False,
    )
    mask = np.zeros((3, 45, 3), dtype=bool)
    mask[:, 18:42, :] = True
    gated = optimise_graph_topology_fixed(
        _two_terminal_pairs(), reconnect_threshold=3.0, validate_reconnections=False,
        mask_support=_mask_support(mask),
    )

    assert "bridge_kind" not in plain.edges["P", "Q", 0]
    assert gated.edges["P", "Q", 0]["bridge_kind"] == "optimise"
    assert gated.edges["P", "Q", 0]["bridge_background_um"] == 0.0


# --- stubs judged by the segmentation ------------------------------------------


def _vessel_with_a_stub(tip):
    """A vessel along x through junction J at (10, 10, 20), and a straight
    stub J -> T."""
    return _graph_from(
        {"A": (10, 10, 0), "J": (10, 10, 20), "B": (10, 10, 40), "T": tip},
        [("A", "J"), ("J", "B"), ("J", "T")],
    )


def _mask_rules(mask):
    from haemolynx.graph.assemble import mask_continues_past

    support = _mask_support(mask)
    return dict(
        inside_lumen=mask_lumen_test(mask, (1.0, 1.0, 1.0)),
        radius_at=lambda p: float(support.radius(np.asarray(p, dtype=float).reshape(1, 3))[0]),
        mask_continues_at=mask_continues_past(support),
    )


def test_a_lee_spur_inside_its_parents_lumen_is_pruned_whatever_its_length():
    """Regression: a 4 um spur from the centreline of a 12 um vessel to its
    wall was kept by a 3 um threshold; its tip is inside the parent's lumen."""
    G = _vessel_with_a_stub((10, 14, 20))
    mask = np.zeros((21, 21, 41), dtype=bool)
    mask[4:17, 4:17, :] = True

    plain = prune_vascular_stubs(G, min_stub_length=3.0)
    judged = prune_vascular_stubs(G, min_stub_length=3.0, **_mask_rules(mask))

    assert "T" in plain
    assert "T" not in judged
    assert {"A", "J", "B"} <= set(judged)


def test_a_stub_mostly_off_the_mask_is_pruned_whatever_its_length():
    G = _vessel_with_a_stub((10, 22, 20))
    mask = np.zeros((21, 25, 41), dtype=bool)
    mask[8:13, 8:13, :] = True

    plain = prune_vascular_stubs(G, min_stub_length=10.0)
    judged = prune_vascular_stubs(G, min_stub_length=10.0, **_mask_rules(mask))

    assert "T" in plain
    assert "T" not in judged


@pytest.mark.parametrize("mask_runs_on", [False, True])
def test_a_short_sprout_is_kept_only_where_the_mask_ends_with_it(mask_runs_on):
    """An 8 um blind sprout (E14.5) ends where its mask does: the radius rule
    keeps it. Where the mask runs on past the tip, the tip is a vessel the
    network lost track of, held to min_stub_length as well, and pruned."""
    G = _vessel_with_a_stub((10, 18, 20))
    mask = np.zeros((21, 34, 41), dtype=bool)
    mask[8:13, 8:13, :] = True
    mask[9:12, 8:(32 if mask_runs_on else 19), 19:22] = True
    settings = dict(min_stub_length=10.0, radius_multiple=1.5)

    rules = _mask_rules(mask)
    plain = prune_vascular_stubs(G, radius_at=rules["radius_at"], **settings)
    judged = prune_vascular_stubs(G, **settings, **rules)

    assert "T" in plain
    assert ("T" in judged) is not mask_runs_on


def test_a_short_stub_at_an_image_face_is_still_kept():
    G = _vessel_with_a_stub((10, 25, 20))
    mask = np.zeros((21, 26, 41), dtype=bool)
    mask[8:13, 8:13, :] = True
    mask[9:12, 8:, 19:22] = True

    judged = prune_vascular_stubs(
        G, min_stub_length=20.0, image_extent_um=(20.0, 25.0, 40.0), **_mask_rules(mask)
    )

    assert "T" in judged
