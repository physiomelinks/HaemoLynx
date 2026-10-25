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
            # An arteriole/venule pair: with no venule mask the masks are global.
            "large_venule_mask_path": "large_venule_mask.tif",
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
            "boundary_handling": "leave_unsolved",
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


def test_assign_boundaries_cuts_and_skips_the_large_tier_when_the_cut_is_on(tmp_path):
    """Ticking the cut back on hides assign_large_vessel_branch_orders in the
    panel (it requires the cut off), so the run cuts and gives no
    Large_Art/Large_Ven tier -- it used to stop on the hidden switch."""
    G, arteriole, venule = _chain_with_large_vessel_masks()
    edges = G.number_of_edges()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(
        plot_dir=tmp_path,
        cut_network_at_large_vessel_volumes=True,
        # This chain leaves no terminal in a mask once cut: pick by hand.
        inlets_outlets_from_vessel_masks=False,
        inlet_node_selection_method="node_ids",
        inlet_node_ids=[3],
        outlet_node_selection_method="node_ids",
        outlet_node_ids=[6],
    )

    boundaries = assign_boundaries(settings, network)

    assert boundaries.graph.number_of_edges() < edges
    assert settings["large_vessel_inlet_nodes"] == []
    assert settings["large_arteriole_boundary_nodes"] == []


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


def test_large_vessel_outlet_override_reads_this_runs_inlet_not_a_stale_one(tmp_path):
    """degree_1_from_inlet measures distance from inlet_nodes -- a bug fixed
    after code review let it read whatever inlet_nodes held over from a
    PREVIOUS run of this same settings dict (the napari GUI reuses one
    across runs), instead of the large-vessel inlet this run just computed.

    settings["inlet_nodes"] is seeded here with a stale value (node 10) that
    is wrong for this run, whose real inlet is node 0 (the arteriole mask's
    own stump). If the stale value leaked through, node 0 -- this run's own
    inlet -- would wrongly qualify as "far enough from the (stale) inlet"
    and be offered as an outlet candidate too.
    """
    G, arteriole, venule = _chain_with_large_vessel_masks_and_extra_terminal()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(
        plot_dir=tmp_path,
        large_vessel_outlet_node_selection_method="degree_1_from_inlet",
        inlet_nodes=[10],
    )

    assign_boundaries(settings, network)

    assert settings["inlet_nodes"] == [0], "this run's real inlet: the arteriole mask's stump"
    # sort_nodes orders by string form ("10" < "9"), hence [10, 9] not [9, 10].
    assert settings["outlet_nodes"] == [10, 9], "far from node 0, not from the stale node 10"
    assert 0 not in settings["outlet_nodes"], "the inlet must never also be offered as an outlet"


def test_large_vessel_inlet_and_outlet_overrides_cannot_collide_on_the_same_node(tmp_path):
    """Two independent overrides that happen to point at the same terminal
    must not both claim it -- exactly like the plain inlet/outlet pair, where
    select_boundary_nodes_for_role's exclude_nodes already prevents this
    (pipeline/stages.py's outlet_nodes = ...exclude_nodes=inlet_nodes). Once
    excluded, the outlet side has nothing left, which correctly raises rather
    than silently duplicating the inlet as the outlet too."""
    G, arteriole, venule = _chain_with_large_vessel_masks_and_extra_terminal()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    settings = _assign_boundaries_settings(
        plot_dir=tmp_path,
        large_vessel_inlet_node_selection_method="coordinates",
        large_vessel_inlet_node_coordinates=[[5.0, 8.0, 5.0]],
        large_vessel_outlet_node_selection_method="coordinates",
        large_vessel_outlet_node_coordinates=[[5.0, 8.0, 5.0]],
    )

    with pytest.raises(ValueError, match="venule mask"):
        assign_boundaries(settings, network)


# --- what is not a stump --------------------------------------------------------------


def test_a_vessel_lying_along_a_face_gives_no_stump_there():
    """On E14.5 the pial arteriole's flank against the top of the stack read
    as a stump: its footprint spans many of its own cross-sections, where a
    vessel cut by a face leaves about one."""
    mask = np.zeros((10, 10, 30), dtype=bool)
    mask[0:3, 2:5, 0:30] = True  # along x, lying on the z=0 face, cut by both x faces

    points = find_large_vessel_mask_stump_points(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert sorted(float(p[2]) for p in points) == [0.0, 29.0], "the two x-face cuts only"
    assert all(float(p[0]) == 1.0 for p in points)


def test_an_oblique_cut_is_still_a_stump():
    """A vessel meeting the face at an angle leaves a longer footprint than
    its cross-section, but not many times longer."""
    mask = np.zeros((12, 12, 12), dtype=bool)
    for z in range(6):
        mask[z, 3:7, 2 * z : 2 * z + 6] = True  # a tube running up at ~27 degrees to the face

    points = find_large_vessel_mask_stump_points(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert any(float(p[0]) == 0.0 for p in points)


def test_specks_below_the_component_fraction_give_no_stump():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0:5, 4:6, 4:6] = True  # the vessel, 20 voxels
    mask[0, 8, 8] = True  # a speck on the same face

    unfiltered = find_large_vessel_mask_stump_points(mask, voxel_size_zyx=(1.0, 1.0, 1.0))
    filtered = find_large_vessel_mask_stump_points(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), min_component_fraction=0.1  # 2 of the vessel's 20 voxels
    )

    assert len(unfiltered) == 2
    assert len(filtered) == 1
    np.testing.assert_allclose(filtered[0], (0.0, 4.5, 4.5))


def test_a_stump_snaps_to_a_terminal_inside_the_mask_not_a_nearer_capillary_tip():
    """A wide vessel's face centroid can lie nearer a capillary tip beside it
    than its own centreline's end; the snap used to take any terminal."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(2.0, 4.5, 4.5))  # the vessel's own end, inside the mask
    G.add_node(1, pos=(5.0, 4.5, 4.5))
    G.add_node(2, pos=(0.0, 4.5, 6.0))  # a capillary tip, outside the mask and nearer the face centroid
    G.add_node(3, pos=(5.0, 8.0, 8.0))
    G.add_edge(0, 1, key=0)
    G.add_edge(1, 3, key=0)
    G.add_edge(2, 3, key=0)
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0:6, 4:6, 4:6] = True

    inlets = select_large_vessel_mask_stump_terminal_nodes_for_role(
        G, mask, node_role="inlet", voxel_size_zyx=(1.0, 1.0, 1.0), image_shape=(10, 10, 10),
        coordinates_setting_name="large_arteriole_mask stump",
    )

    assert inlets == [0]


def test_the_stage_passes_the_component_fraction_setting(tmp_path, monkeypatch):
    import haemolynx.graph.large_vessel_network as module

    G, arteriole, venule = _chain_with_large_vessel_masks()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    seen = []
    real = module._mask_stumps

    def spy(mask, **kwargs):
        seen.append(kwargs.get("min_component_fraction"))
        return real(mask, **kwargs)

    monkeypatch.setattr(module, "_mask_stumps", spy)
    settings = _assign_boundaries_settings(plot_dir=tmp_path, large_vessel_stump_min_component_fraction=0.2)

    assign_boundaries(settings, network)

    assert seen == [0.2, 0.2], "once per side"
    assert settings["inlet_nodes"] == [0] and settings["outlet_nodes"] == [9]
    assert default_schema()["large_vessel_stump_min_component_fraction"].default == 0.05


# --- each stump snaps to its own vessel's end at its face ------------------------


def _path(*corners) -> list[tuple[float, float, float]]:
    """Points every micron along the polyline through *corners*."""
    points: list[tuple[float, float, float]] = []
    for start, end in zip(corners, corners[1:]):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        steps = max(int(np.ceil(np.linalg.norm(end - start))), 1)
        for t in np.linspace(0.0, 1.0, steps + 1)[(1 if points else 0):]:
            points.append(tuple(start + (end - start) * t))
    return points


def _vessel_graph(vessels) -> nx.MultiGraph:
    """One edge per polyline in *vessels* ({(u, v): corners}), its end nodes
    at the polyline's ends."""
    G = nx.MultiGraph()
    for (u, v), corners in vessels.items():
        voxels = _path(*corners)
        G.add_node(u, pos=tuple(float(c) for c in corners[0]))
        G.add_node(v, pos=tuple(float(c) for c in corners[-1]))
        G.add_edge(u, v, voxels=voxels, length=float(len(voxels) - 1))
    return G


def _stump_inlets(G, mask):
    return select_large_vessel_mask_stump_terminal_nodes_for_role(
        G, mask, node_role="inlet", voxel_size_zyx=(1.0, 1.0, 1.0), image_shape=mask.shape,
        coordinates_setting_name="large_arteriole_mask stump",
    )


@pytest.mark.parametrize(
    "tip",
    [(10.0, 16.0, 0.0), (10.0, 10.0, 3.0)],
    ids=["its-end-a-few-um-outside-the-mask-at-the-face", "its-end-just-short-of-the-face"],
)
def test_a_stump_takes_its_own_vessels_end_at_the_face_not_a_terminal_deep_in_the_mask(tip):
    """Regression (E14.5, 2026-10-21): the large masks and the network came
    from two segmentations, so the arteriole's centreline left its mask 4 um
    before the face. Held to the mask, the snap passed that end over for the
    one terminal inside the mask, an arm tip 354 um away."""
    mask = np.zeros((20, 20, 120), dtype=bool)
    mask[7:13, 7:13, 0:100] = True  # a large vessel along x from the x = 0 face
    G = _vessel_graph({
        (0, 1): [tip, (10, 10, 10), (10, 10, 50)],
        (1, 2): [(10, 10, 50), (10, 10, 95)],  # its arm, ending deep inside the mask
        (1, 3): [(10, 10, 50), (10, 19, 50)],  # a capillary leaving it
    })

    assert _stump_inlets(G, mask) == [0]


def test_two_stumps_give_two_nodes_each_its_own_vessels_end():
    """Regression (E14.5): both vein stumps snapped to the one terminal inside
    the vein mask, ~100 um from either, so its two outlets became one."""
    mask = np.zeros((20, 20, 120), dtype=bool)
    mask[7:13, 2:7, 0:50] = True  # vein A, from the x = 0 face
    mask[7:13, 13:18, 0:50] = True  # vein B, beside it, from the same face
    G = _vessel_graph({
        (0, 2): [(10, 8, 0), (10, 4, 8), (10, 4, 45), (10, 10, 60)],  # A's end, 2 um out
        (1, 2): [(10, 12, 0), (10, 15, 8), (10, 15, 45), (10, 10, 60)],  # B's end, 1 um out
        (2, 3): [(10, 10, 60), (10, 10, 110)],
        (4, 5): [(10, 5, 30), (10, 5, 44)],  # a piece lying deep inside A's mask
    })

    outlets = select_large_vessel_mask_stump_terminal_nodes_for_role(
        G, mask, node_role="outlet", voxel_size_zyx=(1.0, 1.0, 1.0), image_shape=mask.shape,
        coordinates_setting_name="large_venule_mask stump",
    )

    assert outlets == [0, 1]


def test_a_capillary_ending_by_the_face_beside_the_large_vessel_is_not_its_end():
    """Its tip is nearer the stump's centroid than the large vessel's own end,
    and as near the mask, but its vessel runs away from the mask."""
    mask = np.zeros((20, 50, 120), dtype=bool)
    mask[7:13, 7:13, 0:100] = True
    G = _vessel_graph({
        (0, 1): [(16, 10, 0), (10, 10, 10), (10, 10, 80)],  # the vessel's end, 3 um out
        (2, 3): [(10, 14, 0), (10, 40, 0)],  # a capillary along the face, its tip 2 um out
    })

    assert _stump_inlets(G, mask) == [0]


def test_a_stump_takes_the_end_whose_vessel_runs_most_along_its_mask():
    """Regression (E14.5, 2026-10-25): a vessel beside the arteriole ran half
    along its mask (54% of its first 50 um) and ended nearer the stump's
    centroid than the arteriole's own end, whose vessel runs all along it;
    the nearer one took the inlet, the Large_Art tier went empty and the
    network resistance rose 140-fold. Both qualify; the one running most
    along the mask is the large vessel's own end."""
    mask = np.zeros((40, 40, 120), dtype=bool)
    mask[17:23, 17:23, 0:100] = True  # the large vessel along x from the x = 0 face
    G = _vessel_graph({
        (0, 1): [(20, 27, 2), (20, 20, 10), (20, 20, 80)],  # its own end, 7 um from the centroid
        (2, 3): [(20, 15, 1), (20, 15, 31), (20, 0, 31), (20, 0, 70)],  # beside it, 5 um from the centroid
    })

    assert _stump_inlets(G, mask) == [0]


def test_a_stump_with_no_vessel_end_at_its_face_warns_how_far_its_node_is():
    """The far-snap warning never fired: it measured the network's extent from
    the candidates alone, one node, so every stump lay outside it."""
    from haemolynx.graph.boundaries import BoundaryCoordinateWarning

    mask = np.zeros((20, 20, 120), dtype=bool)
    mask[7:13, 7:13, 0:100] = True
    G = _vessel_graph({
        (0, 1): [(10, 10, 40), (10, 10, 95)],  # inside the mask, nowhere near the face
        (2, 3): [(10, 19, 0), (10, 19, 30)],  # outside it
    })

    with pytest.warns(BoundaryCoordinateWarning, match=r"x=0 face .* node 0, 40\.\d um away"):
        inlets = _stump_inlets(G, mask)

    assert inlets == [0], "the nearest terminal inside the mask, as before"
