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
