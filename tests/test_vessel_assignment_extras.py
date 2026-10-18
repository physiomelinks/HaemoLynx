"""Tests for hop-distance fallback and disconnected I/O component cleanup."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from haemolynx.graph import (
    FLOW_SOLVED,
    mark_flow_solved_edges,
    remove_components_without_connected_io,
    seed_edges_have_full_mask_coverage,
    select_nodes_at_hop_distance,
)
from haemolynx.pipeline import default_schema


def test_select_nodes_at_hop_distance_exact_ring():
    G = nx.MultiGraph()
    for node in range(5):
        G.add_node(node)
    for node in range(4):
        G.add_edge(node, node + 1)
    assert select_nodes_at_hop_distance(G, [0], 2) == [2]
    assert select_nodes_at_hop_distance(G, [0], 1, exclude_nodes={1}) == []


def test_seed_edges_have_full_mask_coverage_detects_miss():
    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1, voxels=[[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    mask = np.zeros((4, 4, 4), dtype=bool)
    unit = (1.0, 1.0, 1.0)
    covered, uncovered, total = seed_edges_have_full_mask_coverage(G, [0], mask, voxel_size_zyx=unit)
    assert covered is False
    assert uncovered == 1
    assert total == 1
    mask[1, 1, 1] = True
    covered, uncovered, total = seed_edges_have_full_mask_coverage(G, [0], mask, voxel_size_zyx=unit)
    assert covered is True
    assert uncovered == 0
    assert total == 1


def test_seed_coverage_reads_the_mask_through_the_voxel_size():
    """Regression: edge voxels are microns, the mask is voxel-indexed, and the
    check rounded microns straight to indices -- on 2 x 0.5 x 0.5 um voxels an
    edge lying wholly inside the mask read as uncovered, and the fallback
    then threw the mask-found boundary nodes away."""
    from haemolynx.graph import seed_nodes_without_mask_coverage

    voxel_size = (2.0, 0.5, 0.5)
    mask = np.zeros((20, 60, 60), dtype=bool)
    mask[5, 20:40, 20:40] = True  # z index 5 = 10 um; y, x indices 20..39 = 10..20 um
    G = nx.MultiGraph()
    G.add_node(0, pos=(10.0, 15.0, 15.0))
    G.add_node(1, pos=(10.0, 15.0, 18.0))
    G.add_node(2, pos=(30.0, 15.0, 15.0))  # a seed whose edge is outside the mask
    G.add_edge(0, 1, voxels=[(10.0, 15.0, 15.0), (10.0, 15.0, 16.5), (10.0, 15.0, 18.0)])
    G.add_edge(2, 1, voxels=[(30.0, 15.0, 15.0), (20.0, 15.0, 16.5), (10.0, 15.0, 18.0)])

    assert seed_nodes_without_mask_coverage(G, [0, 2], mask, voxel_size_zyx=voxel_size) == []
    G.edges[2, 1, 0]["voxels"] = [(30.0, 15.0, 15.0), (24.0, 15.0, 16.5)]
    assert seed_nodes_without_mask_coverage(G, [0, 2], mask, voxel_size_zyx=voxel_size) == [2]
    covered, uncovered, total = seed_edges_have_full_mask_coverage(
        G, [0], mask, voxel_size_zyx=voxel_size
    )
    assert (covered, uncovered, total) == (True, 0, 1)


def _large_vessel_network_with_small_masks(voxel_size=(2.0, 0.5, 0.5)):
    """A chain along z through a large arteriole (nodes 0-2), small arteriole
    (2-4), capillary (4-5), small venule (5-7) and large venule (7-9), with
    node positions in microns on anisotropic voxels."""
    from haemolynx.pipeline.stages import SkeletonisedVolume, VesselNetwork

    shape = (10, 12, 12)
    scale = np.asarray(voxel_size)
    G = nx.MultiGraph()
    for k in range(shape[0]):
        G.add_node(k, pos=tuple(np.array([k, 5, 5]) * scale))
    for k in range(shape[0] - 1):
        G.add_edge(k, k + 1, voxels=[tuple(np.array([k, 5, 5]) * scale), tuple(np.array([k + 1, 5, 5]) * scale)], length=float(scale[0]))
    large_art = np.zeros(shape, dtype=bool); large_art[0:3, 4:7, 4:7] = True
    small_art = np.zeros(shape, dtype=bool); small_art[2:5, 4:7, 4:7] = True
    small_ven = np.zeros(shape, dtype=bool); small_ven[5:8, 4:7, 4:7] = True
    large_ven = np.zeros(shape, dtype=bool); large_ven[7:10, 4:7, 4:7] = True
    image = np.zeros(shape, dtype=np.uint8)
    volume = SkeletonisedVolume(
        image=image, skeleton=image.copy(),
        voxel_size_xyz=(voxel_size[2], voxel_size[1], voxel_size[0]),
        voxel_size_zyx=tuple(voxel_size), output_dir=Path("."),
    )
    return VesselNetwork(
        graph=G, volume=volume,
        large_arteriole_mask=large_art, large_venule_mask=large_ven,
        small_arteriole_mask=small_art, small_venule_mask=small_ven,
    )


def _large_vessel_mode_settings(tmp_path, **overrides):
    schema = default_schema()
    settings = schema.defaults()
    settings.update({
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "cut_network_at_large_vessel_volumes": False,
        "assign_large_vessel_branch_orders": True,
        "automated_vessel_assignment_fast_mode": False,
        "automated_vessel_assignment_enable_overlap_cleanup": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
        # Arteriole/venule pairs: with no venule mask the masks are global.
        "large_venule_mask_path": "large_venule_mask.tif",
        "small_venule_mask_path": "small_venule_mask.tif",
        "write_small_vessel_boundary_labelling_3d_html": False,
        "visualize_results": False,
        "plot_dir": tmp_path,
        "boundary_handling": "leave_unsolved",
    })
    for name in (
        "inlet_nodes", "outlet_nodes", "arteriole_boundary_nodes", "venule_boundary_nodes",
        "large_arteriole_boundary_nodes", "large_venule_boundary_nodes",
        "inlet_node_coordinates", "outlet_node_coordinates",
        "arteriole_boundary_node_coordinates", "venule_boundary_node_coordinates",
        "arteriole_boundary_node_volumes", "venule_boundary_node_volumes",
    ):
        settings[name] = []
    settings.update(overrides)
    return settings


def test_the_fallback_keeps_the_mask_found_boundary_nodes_when_the_large_vessel_covers_the_inlet(tmp_path):
    """Regression: with the large vessels kept in the network the inlet sits
    on the large arteriole, which the small arteriole mask never covers, so
    the fallback fired on every run and replaced the boundary nodes the
    masks found with the inlet's hop neighbours -- on anisotropic voxels it
    fired for the unit reason too."""
    from haemolynx.graph import infer_boundary_nodes_from_small_vessel_masks
    from haemolynx.pipeline.stages import assign_boundaries

    network = _large_vessel_network_with_small_masks()
    expected = infer_boundary_nodes_from_small_vessel_masks(
        network.graph.copy(), network.small_arteriole_mask, network.small_venule_mask,
        voxel_size_zyx=network.volume.voxel_size_zyx,
    )
    assert len(expected["arteriole_boundary_nodes"]) > 1
    settings = _large_vessel_mode_settings(tmp_path)

    assign_boundaries(settings, network)

    assert settings["inlet_nodes"] == [0] and settings["outlet_nodes"] == [9]
    assert settings["arteriole_boundary_nodes"] == expected["arteriole_boundary_nodes"]
    assert settings["venule_boundary_nodes"] == expected["venule_boundary_nodes"]


def test_the_fallback_adds_hop_nodes_only_for_the_seeds_the_masks_miss(tmp_path):
    """A second inlet whose own vessels no mask reaches gets its hop
    neighbour added; the nodes the masks found stay."""
    from haemolynx.graph import infer_boundary_nodes_from_small_vessel_masks
    from haemolynx.pipeline.stages import assign_boundaries

    network = _large_vessel_network_with_small_masks()
    G = network.graph
    scale = np.asarray(network.volume.voxel_size_zyx)
    # A spur off node 3 ending at a terminal outside every mask, made the
    # second inlet by hand (node_ids), with a node between.
    G.add_node(20, pos=tuple(np.array([3, 5, 9]) * scale))
    G.add_node(21, pos=tuple(np.array([3, 5, 11]) * scale))
    G.add_edge(3, 20, voxels=[tuple(np.array([3, 5, 5]) * scale), tuple(np.array([3, 5, 9]) * scale)], length=2.0)
    G.add_edge(20, 21, voxels=[tuple(np.array([3, 5, 9]) * scale), tuple(np.array([3, 5, 11]) * scale)], length=1.0)
    expected = infer_boundary_nodes_from_small_vessel_masks(
        G.copy(), network.small_arteriole_mask, network.small_venule_mask,
        voxel_size_zyx=network.volume.voxel_size_zyx,
    )
    assert 20 not in expected["arteriole_boundary_nodes"]
    settings = _large_vessel_mode_settings(
        tmp_path,
        inlets_outlets_from_vessel_masks=False,
        inlet_node_selection_method="node_ids",
        inlet_node_ids=[0, 21],
        outlet_node_selection_method="node_ids",
        outlet_node_ids=[9],
    )

    assign_boundaries(settings, network)

    assert settings["inlet_nodes"] == [0, 21]
    assert settings["arteriole_boundary_nodes"] == expected["arteriole_boundary_nodes"] + [20], (
        "the masks' nodes, plus node 21's hop neighbour"
    )
    assert settings["venule_boundary_nodes"] == expected["venule_boundary_nodes"]

    settings = _large_vessel_mode_settings(
        tmp_path,
        inlets_outlets_from_vessel_masks=False,
        inlet_node_selection_method="node_ids",
        inlet_node_ids=[0, 21],
        outlet_node_selection_method="node_ids",
        outlet_node_ids=[9],
        small_vessel_boundary_fallback_to_hop_distance=False,
    )
    assign_boundaries(settings, network)
    assert settings["arteriole_boundary_nodes"] == expected["arteriole_boundary_nodes"], "off: nothing added"


def test_remove_components_without_connected_io_keeps_shared_component():
    G = nx.MultiGraph()
    # Component A: inlet 0 -- 1 -- outlet 2
    G.add_edges_from([(0, 1), (1, 2)])
    # Component B: orphan 10 -- 11
    G.add_edge(10, 11)
    pruned, stats = remove_components_without_connected_io(G, [0], [2])
    assert set(pruned.nodes) == {0, 1, 2}
    assert stats["removed_components"] == 1
    assert stats["removed_nodes"] == 2


def _three_pieces() -> nx.MultiGraph:
    """inlet 0 -- 1 -- outlet 2 (with a parallel 1-2 vessel); inlet 10 -- 11
    with no outlet; 20 -- 21 with no boundary at all."""
    G = nx.MultiGraph()
    G.add_edges_from([(0, 1), (1, 2), (1, 2), (10, 11), (20, 21)])
    return G


def test_mark_flow_solved_edges_marks_what_removal_would_remove():
    G = _three_pieces()
    inlets, outlets = [0, 10], [2]

    unsolved = mark_flow_solved_edges(G, inlets, outlets)

    assert unsolved == 2
    marks = {(u, v, k): data[FLOW_SOLVED] for u, v, k, data in G.edges(keys=True, data=True)}
    assert marks == {
        (0, 1, 0): True, (1, 2, 0): True, (1, 2, 1): True,
        (10, 11, 0): False, (20, 21, 0): False,
    }
    pruned, _stats = remove_components_without_connected_io(G, inlets, outlets)
    kept = {(u, v, k) for u, v, k in pruned.edges(keys=True)}
    assert kept == {edge for edge, solved in marks.items() if solved}


def test_mark_flow_solved_edges_rewrites_an_earlier_mark():
    G = _three_pieces()
    mark_flow_solved_edges(G, [0], [2])
    # An outlet added at 11 makes 10 -- 11 solvable.
    assert mark_flow_solved_edges(G, [0, 10], [2, 11]) == 1
    assert G.edges[10, 11, 0][FLOW_SOLVED] is True
    assert G.edges[20, 21, 0][FLOW_SOLVED] is False


def test_extras_schema_defaults():
    schema = default_schema()
    assert schema["small_vessel_boundary_fallback_to_hop_distance"].default is True
    assert schema["small_vessel_boundary_fallback_hop_distance"].default == 1
    # Removing disconnected components is now a boundary_handling choice
    # (see tests/test_network_handling.py), off by default as the toggle was.
    assert "remove_disconnected_io_components_after_final_assignment" not in schema
    assert schema["boundary_handling"].default == "leave_unsolved"
    assert schema["write_fast_mode_preassignment_large_vessel_debug_3d_html"].default is False
    assert schema["large_vessel_3d_volume_downsample_stride"].default == 1
