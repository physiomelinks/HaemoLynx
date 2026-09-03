"""Empirical tests for cutting graphs at large-vessel mask volumes."""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np

from haemolynx.graph import cut_graph_at_large_vessel_volumes
from haemolynx.gui.results import NODES, VESSELS, ResultLayers, edge_polylines
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.stages import (
    BoundaryNodes,
    SkeletonisedVolume,
    VesselNetwork,
    assign_boundaries,
)


VOXEL_SIZE = (1.0, 1.0, 1.0)


def _point_inside(
    point,
    mask: np.ndarray,
    *,
    voxel_size_zyx=VOXEL_SIZE,
) -> bool:
    idx = tuple(int(round(float(c) / s)) for c, s in zip(point, voxel_size_zyx))
    if any(i < 0 for i in idx):
        return False
    if any(i >= dim for i, dim in zip(idx, mask.shape)):
        return False
    return bool(mask[idx])


def _edge_has_interior_voxel(edge_data: dict, mask: np.ndarray) -> bool:
    voxels = edge_data.get("voxels") or []
    return any(_point_inside(point, mask) for point in voxels)


def _crossing_chain_graph() -> tuple[nx.MultiGraph, np.ndarray, np.ndarray]:
    """One polyline edge that starts outside a cubic volume and ends inside."""
    G = nx.MultiGraph()
    outside = (5.0, 5.0, 0.0)
    inside = (5.0, 5.0, 9.0)
    G.add_node(0, pos=outside)
    G.add_node(1, pos=inside)
    voxels = [(5.0, 5.0, float(x)) for x in range(0, 10)]
    G.add_edge(0, 1, voxels=voxels, length=9.0)

    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[4:7, 4:7, 5:12] = True
    empty = np.zeros_like(mask)
    return G, mask, empty


def _through_volume_orphan_graph() -> tuple[nx.MultiGraph, np.ndarray, np.ndarray]:
    """Main exterior chain + small exterior stub linked only through the volume."""
    G = nx.MultiGraph()
    for node, x in enumerate(range(0, 5)):
        G.add_node(node, pos=(5.0, 5.0, float(x)))
    for node in range(0, 4):
        x0 = float(node)
        x1 = float(node + 1)
        G.add_edge(
            node,
            node + 1,
            voxels=[(5.0, 5.0, x0), (5.0, 5.0, x1)],
            length=1.0,
        )

    G.add_node(5, pos=(5.0, 5.0, 6.0))
    G.add_node(6, pos=(5.0, 5.0, 8.0))
    G.add_edge(
        4,
        5,
        voxels=[(5.0, 5.0, 4.0), (5.0, 5.0, 5.0), (5.0, 5.0, 6.0)],
        length=2.0,
    )
    G.add_edge(
        5,
        6,
        voxels=[(5.0, 5.0, 6.0), (5.0, 5.0, 7.0), (5.0, 5.0, 8.0)],
        length=2.0,
    )

    G.add_node(7, pos=(5.0, 5.0, 10.0))
    G.add_node(8, pos=(5.0, 5.0, 11.0))
    G.add_edge(
        6,
        7,
        voxels=[(5.0, 5.0, 8.0), (5.0, 5.0, 9.0), (5.0, 5.0, 10.0)],
        length=2.0,
    )
    G.add_edge(
        7,
        8,
        voxels=[(5.0, 5.0, 10.0), (5.0, 5.0, 11.0)],
        length=1.0,
    )

    mask = np.zeros((12, 12, 16), dtype=bool)
    mask[4:7, 4:7, 5:10] = True
    empty = np.zeros_like(mask)
    return G, mask, empty


def test_schema_defaults_and_requires_for_volume_cut():
    schema = default_schema()
    assert schema["cut_network_at_large_vessel_volumes"].default is False
    assert schema["remove_orphaned_branches_outside_large_vessel_volumes"].default is False
    assert schema["orphaned_branch_max_edge_count"].default == 3
    assert schema["cut_network_at_large_vessel_volumes"].requires == (
        "use_large_vessel_masks",
        "automated_vessel_assignment",
    )
    assert schema["remove_orphaned_branches_outside_large_vessel_volumes"].requires == (
        "use_large_vessel_masks",
        "automated_vessel_assignment",
        "cut_network_at_large_vessel_volumes",
    )
    assert schema["orphaned_branch_max_edge_count"].requires == (
        "use_large_vessel_masks",
        "automated_vessel_assignment",
        "cut_network_at_large_vessel_volumes",
        "remove_orphaned_branches_outside_large_vessel_volumes",
    )
    assert schema["cut_network_at_large_vessel_volumes"].section == "Vessel masks"


def test_toggle_off_leaves_graph_unchanged():
    G, arteriole, venule = _crossing_chain_graph()
    before_nodes = G.number_of_nodes()
    before_edges = G.number_of_edges()
    before_voxels = [list(d["voxels"]) for _, _, d in G.edges(data=True)]

    result = cut_graph_at_large_vessel_volumes(
        G,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=False,
    )

    assert result.number_of_nodes() == before_nodes
    assert result.number_of_edges() == before_edges
    assert [list(d["voxels"]) for _, _, d in result.edges(data=True)] == before_voxels
    assert set(result.nodes) == set(G.nodes)
    assert nx.is_isomorphic(result, G)


def test_interior_edges_removed_and_cut_creates_degree1_boundary_terminal():
    G, arteriole, venule = _crossing_chain_graph()
    combined = arteriole | venule

    result = cut_graph_at_large_vessel_volumes(
        G,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
        remove_orphaned_branches=False,
    )

    assert result.number_of_edges() == 1
    assert result.number_of_nodes() == 2
    assert 0 in result.nodes
    assert 1 not in result.nodes

    u, v, data = next(iter(result.edges(data=True)))
    assert not _edge_has_interior_voxel(data, combined)

    exterior_voxels = [(5.0, 5.0, float(x)) for x in range(0, 5)]
    assert list(map(tuple, data["voxels"])) == exterior_voxels

    degrees = dict(result.degree())
    assert degrees[0] == 1
    cut_node = v if u == 0 else u
    assert degrees[cut_node] == 1
    assert tuple(result.nodes[cut_node]["pos"]) == (5.0, 5.0, 4.0)


def test_no_remaining_edge_has_interior_voxels():
    G, arteriole, venule = _through_volume_orphan_graph()
    combined = arteriole | venule

    result = cut_graph_at_large_vessel_volumes(
        G,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
        remove_orphaned_branches=False,
    )

    assert result.number_of_edges() == 5
    for _u, _v, data in result.edges(data=True):
        assert not _edge_has_interior_voxel(data, combined)


def test_orphan_cleanup_removes_below_threshold_keeps_above():
    G, arteriole, venule = _through_volume_orphan_graph()

    without_cleanup = cut_graph_at_large_vessel_volumes(
        G,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
        remove_orphaned_branches=False,
    )
    edge_counts = sorted(
        without_cleanup.subgraph(comp).number_of_edges()
        for comp in nx.connected_components(without_cleanup)
    )
    assert edge_counts == [1, 4]

    cleaned = cut_graph_at_large_vessel_volumes(
        G,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
        remove_orphaned_branches=True,
        orphaned_branch_max_edge_count=3,
    )
    assert nx.number_connected_components(cleaned) == 1
    assert cleaned.number_of_edges() == 4
    assert cleaned.number_of_nodes() == 5
    assert 8 not in cleaned.nodes
    assert 7 not in cleaned.nodes
    assert 0 in cleaned.nodes

    kept_at_threshold = cut_graph_at_large_vessel_volumes(
        G,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
        remove_orphaned_branches=True,
        orphaned_branch_max_edge_count=4,
    )
    assert kept_at_threshold.number_of_edges() == 4
    assert nx.number_connected_components(kept_at_threshold) == 1


def test_empty_mask_leaves_topology_unchanged():
    G, arteriole, _venule = _crossing_chain_graph()
    empty = np.zeros_like(arteriole)

    result = cut_graph_at_large_vessel_volumes(
        G,
        empty,
        empty,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )

    assert result.number_of_nodes() == 2
    assert result.number_of_edges() == 1
    assert set(result.nodes) == {0, 1}


def _assign_boundaries_settings(**overrides):
    schema = default_schema()
    settings = schema.defaults()
    settings.update(
        {
            "automated_vessel_assignment": True,
            "use_large_vessel_masks": True,
            "cut_network_at_large_vessel_volumes": True,
            "remove_orphaned_branches_outside_large_vessel_volumes": False,
            "orphaned_branch_max_edge_count": 3,
            "automated_vessel_assignment_fast_mode": False,
            "automated_vessel_assignment_enable_overlap_cleanup": False,
            "automated_vessel_assignment_use_legacy_mode": True,
            "large_vessel_assignment_max_dilation_microns": 0.0,
            "write_fast_mode_preassignment_large_vessel_debug_3d_html": False,
            "use_small_vessel_masks_for_boundary_assignment": False,
            "inlet_nodes": [],
            "outlet_nodes": [],
            "arteriole_boundary_nodes": [],
            "venule_boundary_nodes": [],
            "arteriole_boundary_node_coordinates": [],
            "venule_boundary_node_coordinates": [],
            "arteriole_boundary_node_volumes": [],
            "venule_boundary_node_volumes": [],
            "inlet_node_coordinates": [],
            "outlet_node_coordinates": [],
            "remove_disconnected_io_components_after_final_assignment": False,
        }
    )
    settings.update(overrides)
    return settings


def _network_for_cut(G, arteriole, venule, output_dir) -> VesselNetwork:
    image = np.zeros(arteriole.shape, dtype=np.uint8)
    volume = SkeletonisedVolume(
        image=image,
        skeleton=image.copy(),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=VOXEL_SIZE,
        output_dir=output_dir,
    )
    return VesselNetwork(
        graph=G,
        volume=volume,
        large_arteriole_mask=arteriole,
        large_venule_mask=venule,
    )


def test_assign_boundaries_writes_post_cut_graph_onto_network_and_boundary_nodes(
    tmp_path, monkeypatch
):
    """Downstream must see the cut graph, not a pre-cut copy."""
    G, arteriole, venule = _crossing_chain_graph()
    network = _network_for_cut(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(plot_dir=tmp_path)

    def _fake_legacy_assign(graph_obj, **_kwargs):
        terminals = [n for n, d in graph_obj.degree() if d == 1]
        assert len(terminals) == 2
        return [terminals[0]], [terminals[1]]

    monkeypatch.setattr(
        "haemolynx.graph.select_terminal_nodes_from_large_vessel_masks_progressive_dilation",
        _fake_legacy_assign,
    )
    monkeypatch.setattr(
        "haemolynx.visualization.visualize_3d_plotly_large_vessel_assignment",
        lambda *args, **kwargs: None,
    )

    boundaries = assign_boundaries(settings, network)

    assert boundaries.graph is network.graph
    assert network.graph.number_of_edges() == 1
    assert 1 not in network.graph.nodes
    cut_volume = arteriole | venule
    for _u, _v, data in network.graph.edges(data=True):
        assert not _edge_has_interior_voxel(data, cut_volume)


def test_napari_assign_boundaries_layers_use_post_cut_graph():
    pre_cut, arteriole, venule = _crossing_chain_graph()
    post_cut = cut_graph_at_large_vessel_volumes(
        pre_cut,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )
    assert post_cut.number_of_edges() == 1
    assert post_cut.number_of_nodes() == 2

    results = ResultLayers()
    volume = SkeletonisedVolume(
        image=np.zeros((2, 2, 2), dtype=np.uint8),
        skeleton=np.zeros((2, 2, 2), dtype=np.uint8),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=VOXEL_SIZE,
        output_dir=__import__("pathlib").Path("."),
    )
    results.stage_finished(
        "build_network",
        VesselNetwork(graph=pre_cut, volume=volume),
    )
    cut_terminal = next(n for n in post_cut.nodes if n != 0)
    group = results.stage_finished(
        "assign_boundaries",
        BoundaryNodes(
            inlet_nodes=[0],
            outlet_nodes=[cut_terminal],
            graph=post_cut,
        ),
    )

    vessels = next(spec for spec in group.layers if spec.name == VESSELS)
    nodes = next(spec for spec in group.layers if spec.name == NODES)
    paths, _identity = edge_polylines(post_cut)
    assert len(paths) == 1
    # Vectors layers expand each polyline into consecutive segments.
    assert int(np.unique(vessels.features["edge_index"]).size) == 1
    assert len(nodes.data) == 2
    assert results._graph is post_cut
    assert set(nodes.features["node_id"].tolist()) == set(post_cut.nodes)