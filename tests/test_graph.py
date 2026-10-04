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
    smart_multigraph_degree2_removal,
    merge_edges_with_topology_improvement,
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
    should_add_merged_edge,
)


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


def test_merge_edges_with_topology_improvement():
    v1 = [(0, 0, 0), (1, 0, 0)]
    v2 = [(1, 0, 0), (2, 0, 0)]
    skel = np.zeros((5, 5, 5))
    skel[1, 0, 0] = 1
    out = merge_edges_with_topology_improvement(
        v1, v2, np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([2, 0, 0]), skel
    )
    assert len(out) >= 2


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
    G, _, _ = build_graph_segment_skan_stitched_loops(sk, tiny_skeleton)
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
    G, loops, loop_edges = build_graph_segment_skan_stitched_loops(sk, tiny_skeleton)
    G2, _ = optimise_graph_topology_fixed(
        G, loops, loop_edges, skeleton_data=tiny_skeleton, debug=False
    )
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


def test_build_graph_requires_skan(tiny_skeleton):
    pytest.importorskip("skan")
    from skan import csr
    sk = csr.Skeleton(tiny_skeleton)
    G, loops, loop_edges = build_graph_segment_skan_stitched_loops(
        sk, tiny_skeleton, debug=False
    )
    assert isinstance(G, nx.Graph)
    assert isinstance(loops, list)
    assert isinstance(loop_edges, set)


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

    G, _, _ = build_graph_segment_skan_stitched_loops(
        csr.Skeleton(skeleton), skeleton, reconnect_threshold=12.0
    )
    unguarded, _, _ = build_graph_segment_skan_stitched_loops(
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

    old, _ = optimise_graph_topology_fixed(
        _two_terminal_pairs(), [], set(), reconnect_threshold=3.0,
        validate_reconnections=False, max_bridge_turn_deg=None,
    )
    new, _ = optimise_graph_topology_fixed(
        _two_terminal_pairs(), [], set(), reconnect_threshold=3.0, validate_reconnections=False,
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

    old, _ = optimise_graph_topology_fixed(
        arm_facing_a_vessel_end(), [], set(), reconnect_threshold=3.0,
        validate_reconnections=False, max_bridge_turn_deg=None,
    )
    new, _ = optimise_graph_topology_fixed(
        arm_facing_a_vessel_end(), [], set(), reconnect_threshold=3.0, validate_reconnections=False,
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
