"""Empirical tests for cutting graphs at large-vessel mask volumes."""
from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from haemolynx.graph import cut_graph_at_large_vessel_volumes
from haemolynx.gui.results import NODES, VESSELS, ResultLayers, edge_polylines
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.checks import check_large_vessel_cut_when_masks_enabled
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
    assert schema["cut_network_at_large_vessel_volumes"].default is True
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


def _assert_exterior_kept_interior_removed(result, mask: np.ndarray) -> None:
    """Strict polarity: every remaining edge voxel is outside ``mask``."""
    assert result.number_of_edges() >= 1
    for _u, _v, data in result.edges(data=True):
        assert not _edge_has_interior_voxel(data, mask)
        voxels = list(map(tuple, data["voxels"]))
        assert voxels, "kept edge must retain exterior samples"
        # At least one sample must have been exterior in the original sense.
        assert any(not _point_inside(p, mask) for p in voxels)


def test_arteriole_mask_alone_removes_its_interior_keeps_exterior():
    G, arteriole, empty_venule = _crossing_chain_graph()
    assert not np.any(empty_venule)

    result = cut_graph_at_large_vessel_volumes(
        G,
        arteriole,
        empty_venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )

    _assert_exterior_kept_interior_removed(result, arteriole)
    _u, _v, data = next(iter(result.edges(data=True)))
    assert list(map(tuple, data["voxels"])) == [
        (5.0, 5.0, float(x)) for x in range(0, 5)
    ]
    assert 0 in result.nodes
    assert 1 not in result.nodes


def test_venule_mask_alone_removes_its_interior_keeps_exterior():
    """Venule-only cut must not keep the interior (the reported failure mode)."""
    G, empty_arteriole, _ = _crossing_chain_graph()
    empty_arteriole = np.zeros_like(empty_arteriole)
    venule = np.zeros((12, 12, 12), dtype=bool)
    venule[4:7, 4:7, 5:12] = True
    G.nodes[0]["pos"] = (5.0, 5.0, 0.0)
    G.nodes[1]["pos"] = (5.0, 5.0, 9.0)

    result = cut_graph_at_large_vessel_volumes(
        G,
        empty_arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )

    _assert_exterior_kept_interior_removed(result, venule)
    _u, _v, data = next(iter(result.edges(data=True)))
    kept = list(map(tuple, data["voxels"]))
    assert kept == [(5.0, 5.0, float(x)) for x in range(0, 5)]
    # Inverted polarity would keep x>=5 (interior) instead.
    assert all(p[2] < 5.0 for p in kept)
    assert 1 not in result.nodes


def test_union_of_arteriole_and_venule_masks_removes_both_interiors():
    """Chain crosses arteriole then venule; only exterior gaps remain."""
    G = nx.MultiGraph()
    voxels = [(5.0, 5.0, float(x)) for x in range(0, 20)]
    G.add_node(0, pos=voxels[0])
    G.add_node(1, pos=voxels[-1])
    G.add_edge(0, 1, voxels=voxels, length=19.0)

    arteriole = np.zeros((12, 12, 24), dtype=bool)
    arteriole[4:7, 4:7, 3:7] = True
    venule = np.zeros_like(arteriole)
    venule[4:7, 4:7, 12:16] = True
    combined = arteriole | venule

    result = cut_graph_at_large_vessel_volumes(
        G,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )

    assert result.number_of_edges() == 3
    for _u, _v, data in result.edges(data=True):
        assert not _edge_has_interior_voxel(data, combined)
        assert not _edge_has_interior_voxel(data, arteriole)
        assert not _edge_has_interior_voxel(data, venule)

    kept_x = sorted(
        p[2]
        for _u, _v, data in result.edges(data=True)
        for p in data["voxels"]
    )
    assert kept_x == [0.0, 1.0, 2.0, 7.0, 8.0, 9.0, 10.0, 11.0, 16.0, 17.0, 18.0, 19.0]
    # Interior bands must be gone (inverted polarity would keep these).
    assert not any(3.0 <= x <= 6.0 for x in kept_x)
    assert not any(12.0 <= x <= 15.0 for x in kept_x)


def test_one_two_label_masks_use_minority_foreground_not_all_true_cast():
    """Raw 1/2-encoded masks must not become all-True via dtype=bool."""
    G, arteriole_bool, empty = _crossing_chain_graph()
    # Encode True voxels as label 1, background as label 2 (ilastik-style).
    arteriole_12 = np.full(arteriole_bool.shape, 2, dtype=np.uint8)
    arteriole_12[arteriole_bool] = 1
    assert np.asarray(arteriole_12, dtype=bool).all()  # naive cast is wrong

    result = cut_graph_at_large_vessel_volumes(
        G,
        arteriole_12,
        np.full(arteriole_bool.shape, 2, dtype=np.uint8),
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )
    _assert_exterior_kept_interior_removed(result, arteriole_bool)
    assert result.number_of_edges() == 1
    assert 1 not in result.nodes


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


def test_venule_shaped_input_network_cut_keeps_only_exterior_samples():
    """Hand-traced case matching the user symptom configuration.

    When the main network was skeletonised from a venule-like volume (every
    centreline starts inside the venule mask and exits briefly), exterior-keep
    must still discard interior samples. Inverted polarity would retain the
    interior band and drop the exterior tip — the reported failure mode.
    """
    G = nx.MultiGraph()
    # One edge fully inside venule, one that exits to exterior.
    G.add_node(0, pos=(5.0, 5.0, 6.0))
    G.add_node(1, pos=(5.0, 5.0, 9.0))
    G.add_edge(
        0,
        1,
        voxels=[(5.0, 5.0, float(x)) for x in range(6, 10)],
        length=3.0,
    )
    G.add_node(2, pos=(5.0, 5.0, 0.0))
    G.add_node(3, pos=(5.0, 5.0, 7.0))
    G.add_edge(
        2,
        3,
        voxels=[(5.0, 5.0, float(x)) for x in range(0, 8)],
        length=7.0,
    )

    arteriole = np.zeros((12, 12, 12), dtype=bool)
    # Off-axis arteriole so it does not intersect the test polyline; still
    # participates in the cut volume union.
    arteriole[0:2, 0:2, 0:2] = True
    venule = np.zeros((12, 12, 12), dtype=bool)
    venule[4:7, 4:7, 5:12] = True
    combined = arteriole | venule

    result = cut_graph_at_large_vessel_volumes(
        G,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )

    # Fully interior edge gone; crossing edge keeps only x < 5 (exterior).
    assert result.number_of_edges() == 1
    _u, _v, data = next(iter(result.edges(data=True)))
    kept = list(map(tuple, data["voxels"]))
    assert kept == [(5.0, 5.0, float(x)) for x in range(0, 5)]
    assert all(not _point_inside(p, combined) for p in kept)
    assert all(not _point_inside(p, venule) for p in kept)
    assert all(not _point_inside(p, arteriole) for p in kept)
    # Inverted polarity would have kept the venule interior band instead.
    assert not any(p[2] >= 5.0 for p in kept)
    assert 0 not in result.nodes and 1 not in result.nodes


def test_assign_boundaries_passes_both_arteriole_and_venule_masks_to_cut(
    tmp_path, monkeypatch
):
    """Cut must receive both masks (union), not venule alone."""
    G, arteriole, venule = _crossing_chain_graph()
    # Distinct arteriole pocket so we can assert both arrays were forwarded.
    arteriole = np.zeros_like(arteriole)
    arteriole[1:3, 1:3, 1:3] = True
    venule = np.zeros_like(venule)
    venule[4:7, 4:7, 5:12] = True
    network = _network_for_cut(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(plot_dir=tmp_path)
    captured: dict = {}

    def _capture_cut(graph_obj, art_mask, ven_mask, **kwargs):
        captured["arteriole"] = np.asarray(art_mask)
        captured["venule"] = np.asarray(ven_mask)
        return cut_graph_at_large_vessel_volumes(
            graph_obj, art_mask, ven_mask, **kwargs
        )

    monkeypatch.setattr(
        "haemolynx.graph.cut_graph_at_large_vessel_volumes",
        _capture_cut,
    )
    monkeypatch.setattr(
        "haemolynx.graph.select_terminal_nodes_from_large_vessel_masks_progressive_dilation",
        lambda graph_obj, **_kwargs: (
            [n for n, d in graph_obj.degree() if d == 1][:1],
            [n for n, d in graph_obj.degree() if d == 1][1:2],
        ),
    )
    monkeypatch.setattr(
        "haemolynx.visualization.visualize_3d_plotly_large_vessel_assignment",
        lambda *args, **kwargs: None,
    )

    assign_boundaries(settings, network)

    assert "arteriole" in captured and "venule" in captured
    assert np.array_equal(captured["arteriole"], arteriole)
    assert np.array_equal(captured["venule"], venule)
    assert np.any(captured["arteriole"])
    assert np.any(captured["venule"])


def test_preflight_warns_when_input_path_is_the_large_venule_mask(tmp_path):
    from haemolynx.pipeline.checks import check_input_is_not_a_large_vessel_mask

    venule = tmp_path / "HaemoLynx_large_venule_mask.tif"
    arteriole = tmp_path / "large_arteriole_mask.tif"
    venule.write_bytes(b"x")
    arteriole.write_bytes(b"y")
    report = check_input_is_not_a_large_vessel_mask(
        {
            "use_ilastik_segmentation": False,
            "use_large_vessel_masks": True,
            "cut_network_at_large_vessel_volumes": True,
            "input_path": str(venule),
            "large_arteriole_mask_path": str(arteriole),
            "large_venule_mask_path": str(venule),
        }
    )
    assert report.warnings
    assert any("venule" in w.lower() for w in report.warnings)
    assert any("input_path" in w for w in report.warnings)


def test_napari_empty_post_cut_graph_emits_empty_vessel_and_node_layers():
    """An empty post-cut graph must clear pre-cut geometry, not omit layer specs."""
    pre_cut, arteriole, venule = _crossing_chain_graph()
    # Fully interior chain: cut removes everything.
    for node in pre_cut.nodes:
        z, y, x = pre_cut.nodes[node]["pos"]
        pre_cut.nodes[node]["pos"] = (z, y, x + 6.0)
    for _u, _v, data in pre_cut.edges(data=True):
        data["voxels"] = [(5.0, 5.0, float(x)) for x in range(6, 10)]

    post_cut = cut_graph_at_large_vessel_volumes(
        pre_cut,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )
    assert post_cut.number_of_edges() == 0
    assert post_cut.number_of_nodes() == 0

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
    group = results.stage_finished(
        "assign_boundaries",
        BoundaryNodes(inlet_nodes=[], outlet_nodes=[], graph=post_cut),
    )

    vessels = next(spec for spec in group.layers if spec.name == VESSELS)
    nodes = next(spec for spec in group.layers if spec.name == NODES)
    assert len(np.asarray(vessels.data)) == 0
    assert len(np.asarray(nodes.data)) == 0
    assert results._graph is post_cut


def test_default_schema_cuts_interior_edges_with_large_mask_assignment(
    tmp_path, monkeypatch
):
    """Large-mask assignment with schema defaults must cut without a second opt-in."""
    G, arteriole, venule = _crossing_chain_graph()
    network = _network_for_cut(G, arteriole, venule, tmp_path)
    schema = default_schema()
    settings = schema.defaults()
    settings.update(
        {
            "input_path": tmp_path / "stack.tif",
            "plot_dir": tmp_path,
            "automated_vessel_assignment": True,
            "use_large_vessel_masks": True,
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
    assert settings["cut_network_at_large_vessel_volumes"] is True

    monkeypatch.setattr(
        "haemolynx.graph.select_terminal_nodes_from_large_vessel_masks_progressive_dilation",
        lambda graph_obj, **_kwargs: (
            [n for n, d in graph_obj.degree() if d == 1][:1],
            [n for n, d in graph_obj.degree() if d == 1][1:2],
        ),
    )
    monkeypatch.setattr(
        "haemolynx.visualization.visualize_3d_plotly_large_vessel_assignment",
        lambda *args, **kwargs: None,
    )

    assign_boundaries(settings, network)

    cut_volume = arteriole | venule
    assert network.graph.number_of_edges() == 1
    assert 1 not in network.graph.nodes
    for _u, _v, data in network.graph.edges(data=True):
        assert not _edge_has_interior_voxel(data, cut_volume)


def test_assign_boundaries_rewrites_graph_pickle_after_cut(tmp_path, monkeypatch):
    G, arteriole, venule = _crossing_chain_graph()
    stem = "stack"
    input_path = tmp_path / f"{stem}.tif"
    input_path.write_bytes(b"x")
    network = _network_for_cut(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(
        plot_dir=tmp_path,
        input_path=str(input_path),
    )
    graph_path = tmp_path / f"{stem}_graph.pkl"
    with graph_path.open("wb") as handle:
        pickle.dump(G, handle)
    assert G.number_of_edges() == 1

    monkeypatch.setattr(
        "haemolynx.graph.select_terminal_nodes_from_large_vessel_masks_progressive_dilation",
        lambda graph_obj, **_kwargs: (
            [n for n, d in graph_obj.degree() if d == 1][:1],
            [n for n, d in graph_obj.degree() if d == 1][1:2],
        ),
    )
    monkeypatch.setattr(
        "haemolynx.visualization.visualize_3d_plotly_large_vessel_assignment",
        lambda *args, **kwargs: None,
    )

    assign_boundaries(settings, network)

    with graph_path.open("rb") as handle:
        restored = pickle.load(handle)
    assert restored.number_of_edges() == network.graph.number_of_edges()
    assert restored.number_of_edges() == 1
    assert set(restored.nodes) == set(network.graph.nodes)


def test_preflight_warns_when_large_masks_on_and_cut_off():
    report = check_large_vessel_cut_when_masks_enabled(
        {
            "use_large_vessel_masks": True,
            "automated_vessel_assignment": True,
            "cut_network_at_large_vessel_volumes": False,
        }
    )
    assert report.warnings
    assert any("cut_network_at_large_vessel_volumes" in warning for warning in report.warnings)

    clear = check_large_vessel_cut_when_masks_enabled(
        {
            "use_large_vessel_masks": True,
            "automated_vessel_assignment": True,
            "cut_network_at_large_vessel_volumes": True,
        }
    )
    assert not clear.warnings


def test_sparse_chord_through_mask_is_not_kept_whole():
    """Two-endpoint edges must not skip a mask volume they pass through."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(5.0, 5.0, 0.0))
    G.add_node(1, pos=(5.0, 5.0, 12.0))
    G.add_edge(
        0,
        1,
        voxels=[(5.0, 5.0, 0.0), (5.0, 5.0, 12.0)],
        length=12.0,
    )
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[4:7, 4:7, 5:12] = True
    combined = mask

    result = cut_graph_at_large_vessel_volumes(
        G,
        mask,
        np.zeros_like(mask),
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )

    for _u, _v, data in result.edges(data=True):
        assert not _edge_has_interior_voxel(data, combined)


def test_napari_solve_layers_use_solved_post_cut_graph():
    """Solve must rebuild vessels from the solved graph, not a stale pre-cut copy."""
    from haemolynx.pipeline.stages import HaemodynamicModel, Solution

    pre_cut, arteriole, venule = _crossing_chain_graph()
    post_cut = cut_graph_at_large_vessel_volumes(
        pre_cut,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )
    cut_terminal = next(n for n in post_cut.nodes if n != 0)
    volume = SkeletonisedVolume(
        image=np.zeros((2, 2, 2), dtype=np.uint8),
        skeleton=np.zeros((2, 2, 2), dtype=np.uint8),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=VOXEL_SIZE,
        output_dir=Path("."),
    )

    results = ResultLayers()
    results.stage_finished(
        "build_network",
        VesselNetwork(graph=pre_cut, volume=volume),
    )
    results.stage_finished(
        "assign_boundaries",
        BoundaryNodes(
            inlet_nodes=[0],
            outlet_nodes=[cut_terminal],
            graph=post_cut,
        ),
    )
    # Simulate a stale remembered graph (pre-cut) from an intermediate stage.
    results._graph = pre_cut
    results.stage_finished(
        "assign_diameters",
        HaemodynamicModel(graph=pre_cut),
    )
    solve_group = results.stage_finished(
        "solve",
        Solution(
            graph=post_cut,
            pressure=np.zeros(post_cut.number_of_nodes()),
            node_list=list(post_cut.nodes),
            equivalent_resistance=1.0,
        ),
    )

    vessels = next(spec for spec in solve_group.layers if spec.name == VESSELS)
    assert results._graph is post_cut
    assert int(np.unique(vessels.features["edge_index"]).size) == post_cut.number_of_edges()
    paths, _identity = edge_polylines(post_cut)
    assert len(paths) == post_cut.number_of_edges()
    combined = arteriole | venule
    for path in paths:
        assert not any(_point_inside(tuple(point), combined) for point in path)


def test_from_solve_keeps_skeleton_hidden():
    from haemolynx.pipeline.stages import Solution

    results = ResultLayers()
    skeleton = np.zeros((4, 4, 4), dtype=np.uint8)
    results._skeleton = skeleton
    graph = _crossing_chain_graph()[0]
    results.stage_finished(
        "build_network",
        VesselNetwork(
            graph=graph,
            volume=SkeletonisedVolume(
                image=np.zeros((4, 4, 4), dtype=np.uint8),
                skeleton=skeleton,
                voxel_size_xyz=(1.0, 1.0, 1.0),
                voxel_size_zyx=VOXEL_SIZE,
                output_dir=Path("."),
            ),
        ),
    )
    solve_group = results.stage_finished(
        "solve",
        Solution(
            graph=graph,
            pressure=np.zeros(graph.number_of_nodes()),
            node_list=list(graph.nodes),
            equivalent_resistance=1.0,
        ),
    )
    from haemolynx.gui.results import SKELETON

    skeleton_spec = next(spec for spec in solve_group.layers if spec.name == SKELETON)
    assert skeleton_spec.visible is False


def _post_cut_pipeline_graphs():
    """Pre-cut chain, masks, and the cut graph assign_boundaries should keep."""
    pre_cut, arteriole, venule = _crossing_chain_graph()
    post_cut = cut_graph_at_large_vessel_volumes(
        pre_cut,
        arteriole,
        venule,
        voxel_size_zyx=VOXEL_SIZE,
        enabled=True,
    )
    return pre_cut, post_cut, arteriole, venule


def test_post_cut_graph_identity_through_diameters_and_solve(tmp_path, monkeypatch):
    """One MultiGraph object from cut through diameters and solve."""
    from haemolynx.pipeline.stages import HaemodynamicModel, assign_diameters, solve

    pre_cut, post_cut, arteriole, venule = _post_cut_pipeline_graphs()
    network = _network_for_cut(pre_cut, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(plot_dir=tmp_path)
    settings["inlet_nodes"] = [0]
    settings["outlet_nodes"] = [next(n for n in post_cut.nodes if n != 0)]
    settings["run_haemodynamics"] = False

    monkeypatch.setattr(
        "haemolynx.graph.select_terminal_nodes_from_large_vessel_masks_progressive_dilation",
        lambda graph_obj, **_kwargs: (
            [n for n, d in graph_obj.degree() if d == 1][:1],
            [n for n, d in graph_obj.degree() if d == 1][1:2],
        ),
    )
    monkeypatch.setattr(
        "haemolynx.visualization.visualize_3d_plotly_large_vessel_assignment",
        lambda *args, **kwargs: None,
    )

    boundaries = assign_boundaries(settings, network)
    assert boundaries.graph is network.graph
    assert network.graph is not pre_cut
    assert network.graph.number_of_edges() == post_cut.number_of_edges()

    model = assign_diameters(settings, network, boundaries, default_schema())
    assert model.graph is network.graph
    assert model.graph is boundaries.graph

    settings["run_haemodynamics"] = True
    settings["do_equiv_resistance_calculation"] = False
    model.graph = network.graph
    solution = solve(settings, model, boundaries)
    assert solution.graph is network.graph
    assert set(solution.graph.edges()) == set(post_cut.edges())


def test_napari_assign_diameters_does_not_restore_pre_cut_vessels():
    """A stale pre-cut graph on the diameters output must not regrow interior edges."""
    from haemolynx.pipeline.stages import HaemodynamicModel

    pre_cut, post_cut, _arteriole, _venule = _post_cut_pipeline_graphs()
    volume = SkeletonisedVolume(
        image=np.zeros((2, 2, 2), dtype=np.uint8),
        skeleton=np.zeros((2, 2, 2), dtype=np.uint8),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=VOXEL_SIZE,
        output_dir=Path("."),
    )
    results = ResultLayers()
    results.stage_finished("build_network", VesselNetwork(graph=pre_cut, volume=volume))
    cut_terminal = next(n for n in post_cut.nodes if n != 0)
    results.stage_finished(
        "assign_boundaries",
        BoundaryNodes(
            inlet_nodes=[0],
            outlet_nodes=[cut_terminal],
            graph=post_cut,
        ),
    )
    results._graph = pre_cut  # simulate a stale remembered graph

    group = results.stage_finished(
        "assign_diameters",
        HaemodynamicModel(graph=pre_cut),
    )
    vessels = next(spec for spec in group.layers if spec.name == VESSELS)
    assert results._graph is post_cut
    assert int(np.unique(vessels.features["edge_index"]).size) == post_cut.number_of_edges()


def test_napari_build_haemodynamic_model_does_not_restore_pre_cut_vessels():
    """Recolour-only stage must not revert vessel geometry to a pre-cut copy."""
    from haemolynx.pipeline.stages import HaemodynamicModel

    pre_cut, post_cut, _arteriole, _venule = _post_cut_pipeline_graphs()
    volume = SkeletonisedVolume(
        image=np.zeros((2, 2, 2), dtype=np.uint8),
        skeleton=np.zeros((2, 2, 2), dtype=np.uint8),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=VOXEL_SIZE,
        output_dir=Path("."),
    )
    results = ResultLayers()
    results.stage_finished("build_network", VesselNetwork(graph=pre_cut, volume=volume))
    cut_terminal = next(n for n in post_cut.nodes if n != 0)
    results.stage_finished(
        "assign_boundaries",
        BoundaryNodes(
            inlet_nodes=[0],
            outlet_nodes=[cut_terminal],
            graph=post_cut,
        ),
    )
    results.stage_finished(
        "assign_diameters",
        HaemodynamicModel(graph=post_cut),
    )
    results._graph = pre_cut

    results.stage_finished(
        "build_haemodynamic_model",
        HaemodynamicModel(graph=pre_cut),
    )
    assert results._graph is post_cut