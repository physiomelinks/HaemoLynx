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
    assign_branch_orders,
    select_boundary_nodes_by_method,
    select_terminal_nodes_from_large_vessel_masks,
    infer_boundary_nodes_from_small_vessel_masks,
    diagnose_degree2_nodes,
    format_degree2_diagnostics_report,
)
from ImageLynx.graph._helpers import (
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
    e1 = {"voxels": [(0, 0, 0), (1, 0, 0)], "weight": 1, "length": 1}
    e2 = {"voxels": [(1, 0, 0), (2, 0, 0)], "weight": 1, "length": 1}
    merged = create_trivial_merged_edge(e1, e2, np.array([1, 0, 0]))
    assert merged["weight"] == 2
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


def test_safer_simple_remove_all_degree2_nodes(simple_graph):
    G = safer_simple_remove_all_degree2_nodes(simple_graph.copy(), max_degree=5)
    assert G.number_of_nodes() <= 3


def test_trivial_remove_all_degree2_nodes(simple_graph):
    G = trivial_remove_all_degree2_nodes(simple_graph.copy(), max_degree=5)
    assert G.number_of_nodes() <= 3


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


def test_smart_multigraph_degree2_removal(simple_graph):
    G = nx.MultiGraph(simple_graph)
    G2 = smart_multigraph_degree2_removal(G, skeleton_data=None, debug=False)
    assert isinstance(G2, nx.MultiGraph)


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
        voxel_size_xyz=(1.0, 1.0, 1.0),
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
        voxel_size_xyz=(1.0, 1.0, 1.0),
        allow_overlap=False,
    )
    assert start_nodes == [0]
    assert out_nodes == [1]


def test_infer_boundary_nodes_from_small_vessel_masks():
    G = nx.MultiGraph()
    for node_id, pos in {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (3.0, 0.0, 0.0),
        4: (4.0, 0.0, 0.0),
        5: (5.0, 0.0, 0.0),
    }.items():
        G.add_node(node_id, pos=np.asarray(pos, dtype=float))

    G.add_edge(0, 1, voxels=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], length=1.0, weight=1.0)
    G.add_edge(1, 2, voxels=[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], length=1.0, weight=1.0)
    G.add_edge(2, 3, voxels=[(2.0, 0.0, 0.0), (3.0, 0.0, 0.0)], length=1.0, weight=1.0)
    G.add_edge(3, 4, voxels=[(3.0, 0.0, 0.0), (4.0, 0.0, 0.0)], length=1.0, weight=1.0)
    G.add_edge(4, 5, voxels=[(4.0, 0.0, 0.0), (5.0, 0.0, 0.0)], length=1.0, weight=1.0)

    small_arteriole_mask = np.zeros((8, 4, 8), dtype=bool)
    small_venule_mask = np.zeros((8, 4, 8), dtype=bool)
    small_arteriole_mask[0:2, 0, 0] = True
    small_venule_mask[4:6, 0, 0] = True

    result = infer_boundary_nodes_from_small_vessel_masks(
        G,
        small_arteriole_mask=small_arteriole_mask,
        small_venule_mask=small_venule_mask,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        minimum_overlap_fraction=0.5,
    )

    assert result["arteriole_boundary_nodes"] == [2]
    assert result["venule_boundary_nodes"] == [3]
    assert result["arteriole_edge_count"] == 2
    assert result["venule_edge_count"] == 2
    assert G.nodes[2]["mask_vessel_type"] == "arteriole"
    assert G.nodes[3]["mask_vessel_type"] == "venule"

