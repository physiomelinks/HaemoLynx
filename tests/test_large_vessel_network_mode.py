"""Large vessels kept in the network as Large_Art/Large_Ven branch orders."""
from __future__ import annotations

import numpy as np
import pytest
import networkx as nx

from haemolynx.graph.large_vessel_network import (
    find_large_vessel_mask_stump_points,
    select_large_vessel_mask_stump_terminal_nodes_for_role,
    select_large_vessel_stump_terminal_nodes,
)
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.stages import (
    SkeletonisedVolume,
    VesselNetwork,
    assign_boundaries,
)


def test_find_large_vessel_mask_stump_points_finds_face_touching_component_only():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0:3, 4:6, 4:6] = True  # touches z=0
    mask[6:9, 6:8, 6:8] = True  # fully interior, no face contact

    points = find_large_vessel_mask_stump_points(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert len(points) == 1
    np.testing.assert_allclose(points[0], (0.0, 4.5, 4.5))


def test_find_large_vessel_mask_stump_points_handles_two_separate_stumps():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0:2, 1:3, 1:3] = True  # touches z=0, away from the other component
    mask[0:2, 7:9, 7:9] = True  # also touches z=0, but a separate component

    points = find_large_vessel_mask_stump_points(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert len(points) == 2
    z_values = sorted(float(p[0]) for p in points)
    assert z_values == [0.0, 0.0]
    y_values = sorted(float(p[1]) for p in points)
    assert y_values == [1.5, 7.5]


def test_find_large_vessel_mask_stump_points_empty_mask_returns_empty_list():
    assert find_large_vessel_mask_stump_points(
        np.zeros((5, 5, 5), dtype=bool), voxel_size_zyx=(1.0, 1.0, 1.0)
    ) == []


def _straight_chain_graph(length: int) -> nx.MultiGraph:
    """A straight chain of *length* nodes along z at (5, 5, z)."""
    G = nx.MultiGraph()
    for z in range(length):
        G.add_node(z, pos=(float(z), 5.0, 5.0))
    for z in range(length - 1):
        G.add_edge(
            z,
            z + 1,
            voxels=[(float(z), 5.0, 5.0), (float(z + 1), 5.0, 5.0)],
            length=1.0,
        )
    return G


def test_select_large_vessel_stump_terminal_nodes_ignores_interior_dead_end():
    """The scenario the whole module exists for: two degree-1 nodes sit
    inside the arteriole mask's footprint, but only one of them -- the one
    at z=0 -- actually touches the image's own face. Overlap-based terminal
    selection cannot distinguish these; this must."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 5.0, 5.0))  # true image-edge stump
    G.add_node(1, pos=(3.0, 5.0, 5.0))  # branch point
    G.add_node(2, pos=(3.0, 9.0, 5.0))  # interior arm tip, degree-1, no face contact
    G.add_edge(0, 1, key=0)
    G.add_edge(1, 2, key=0)

    arteriole_mask = np.zeros((10, 10, 10), dtype=bool)
    # Touches only the z=0 face -- deliberately not reaching node 2's y=9
    # column, so this test isolates "picks the face-touching stump" from
    # any question of which nodes happen to overlap the mask at all.
    arteriole_mask[0:5, 4:6, 4:6] = True
    venule_mask = np.zeros((10, 10, 10), dtype=bool)

    inlets, outlets = select_large_vessel_stump_terminal_nodes(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        image_shape=(10, 10, 10),
    )

    assert inlets == [0]
    assert outlets == []


def test_select_large_vessel_mask_stump_terminal_nodes_for_role_matches_the_two_sided_call():
    """select_large_vessel_stump_terminal_nodes is now a thin wrapper calling
    this once per side -- pinned equal for one side, on the same fixture."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 5.0, 5.0))
    G.add_node(1, pos=(3.0, 5.0, 5.0))
    G.add_node(2, pos=(3.0, 9.0, 5.0))
    G.add_edge(0, 1, key=0)
    G.add_edge(1, 2, key=0)

    arteriole_mask = np.zeros((10, 10, 10), dtype=bool)
    arteriole_mask[0:5, 4:6, 4:6] = True

    inlets = select_large_vessel_mask_stump_terminal_nodes_for_role(
        G,
        arteriole_mask,
        node_role="inlet",
        voxel_size_zyx=(1.0, 1.0, 1.0),
        image_shape=(10, 10, 10),
        coordinates_setting_name="large_arteriole_mask stump",
    )

    assert inlets == [0]


def _assign_boundaries_settings(**overrides):
    schema = default_schema()
    settings = schema.defaults()
    settings.update(
        {
            "automated_vessel_assignment": True,
            "use_large_vessel_masks": True,
            "use_thick_vessel_skeletonisation": True,
            "cut_network_at_large_vessel_volumes": False,
            "assign_large_vessel_branch_orders": True,
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
            "large_arteriole_boundary_nodes": [],
            "large_venule_boundary_nodes": [],
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


def _network_for_large_vessel_mode(G, arteriole, venule, output_dir) -> VesselNetwork:
    image = np.zeros(arteriole.shape, dtype=np.uint8)
    volume = SkeletonisedVolume(
        image=image,
        skeleton=image.copy(),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=output_dir,
    )
    return VesselNetwork(
        graph=G,
        volume=volume,
        large_arteriole_mask=arteriole,
        large_venule_mask=venule,
    )


def _chain_with_large_vessel_masks():
    """A 10-node straight chain spanning the whole (10, 10, 10) volume; the
    first 3 edges sit in an arteriole mask touching the z=0 face, the last
    3 in a venule mask touching the z=9 face."""
    shape = (10, 10, 10)
    G = _straight_chain_graph(shape[0])
    arteriole = np.zeros(shape, dtype=bool)
    arteriole[0:3, 4:6, 4:6] = True  # touches z=0; covers nodes 0, 1, 2
    venule = np.zeros(shape, dtype=bool)
    venule[7:10, 4:6, 4:6] = True  # touches z=9; covers nodes 7, 8, 9
    return G, arteriole, venule


def test_assign_boundaries_large_vessel_network_mode_populates_large_boundary_node_settings(
    tmp_path,
):
    G, arteriole, venule = _chain_with_large_vessel_masks()
    before_edges = G.number_of_edges()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(plot_dir=tmp_path)

    boundaries = assign_boundaries(settings, network)

    # Not cut: every original edge (and its interior-mask voxels) survives.
    assert boundaries.graph.number_of_edges() == before_edges
    assert settings["inlet_nodes"] == [0]
    assert settings["large_vessel_inlet_nodes"] == [0], "mask_stump is the default method"
    assert settings["large_arteriole_boundary_nodes"]
    assert settings["large_venule_boundary_nodes"]


def test_assign_boundaries_raises_when_large_vessel_mode_and_cut_both_on(tmp_path):
    G, arteriole, venule = _chain_with_large_vessel_masks()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(
        plot_dir=tmp_path, cut_network_at_large_vessel_volumes=True
    )

    with pytest.raises(ValueError, match="cut_network_at_large_vessel_volumes"):
        assign_boundaries(settings, network)


def test_assign_boundaries_logs_large_vessel_network_behaviour_change(tmp_path, caplog):
    G, arteriole, venule = _chain_with_large_vessel_masks()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(plot_dir=tmp_path)

    with caplog.at_level("INFO", logger="haemolynx.pipeline.stages"):
        assign_boundaries(settings, network)

    assert any("Large-vessel network mode" in record.message for record in caplog.records)


# --- large_vessel_inlet/outlet: mask_stump default vs. a manual override ----


def _chain_with_large_vessel_masks_and_extra_terminal():
    """Like _chain_with_large_vessel_masks, but with a spur off node 5 ending
    at a degree-1 node that touches neither mask -- a candidate a manual
    override can point at, distinct from the arteriole mask's own stump."""
    G, arteriole, venule = _chain_with_large_vessel_masks()
    G.add_node(10, pos=(5.0, 8.0, 5.0))
    G.add_edge(5, 10, key=0, voxels=[(5.0, 5.0, 5.0), (5.0, 8.0, 5.0)], length=3.0)
    return G, arteriole, venule


def test_large_vessel_inlet_override_replaces_the_mask_stump(tmp_path):
    """large_vessel_inlet_node_selection_method defaults to mask_stump; the
    other four boundary roles' methods (coordinates here) override it with a
    manual pick, exactly as for inlet/outlet/arteriole_boundary/venule_boundary."""
    G, arteriole, venule = _chain_with_large_vessel_masks_and_extra_terminal()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(
        plot_dir=tmp_path,
        large_vessel_inlet_node_selection_method="coordinates",
        large_vessel_inlet_node_coordinates=[[5.0, 8.0, 5.0]],
    )

    assign_boundaries(settings, network)

    assert settings["inlet_nodes"] == [10], "the override, not the mask's own stump (node 0)"
    assert settings["large_vessel_inlet_nodes"] == [10]


def test_large_vessel_outlet_still_falls_back_to_mask_stump_when_only_the_inlet_is_overridden(
    tmp_path,
):
    G, arteriole, venule = _chain_with_large_vessel_masks_and_extra_terminal()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(
        plot_dir=tmp_path,
        large_vessel_inlet_node_selection_method="coordinates",
        large_vessel_inlet_node_coordinates=[[5.0, 8.0, 5.0]],
    )

    assign_boundaries(settings, network)

    assert settings["outlet_nodes"] == [9], "untouched: still the venule mask's own stump"
    assert settings["large_vessel_outlet_nodes"] == [9]
