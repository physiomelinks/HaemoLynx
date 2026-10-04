"""Keeping the large-vessel masks while choosing the inlets and outlets by hand.

automated_vessel_assignment used to do two jobs at once: drive the
large-vessel treatment from the arteriole/venule masks, and take over the
inlets and outlets -- greying the Inlet and Outlet tabs, and overwriting
whatever was set there. inlets_outlets_from_vessel_masks now holds the second
job alone, so a run can keep its masks and set its inlets and outlets.
"""
from __future__ import annotations

import pytest

from haemolynx.graph import inlets_outlets_from_vessel_masks
from haemolynx.gui.boundary_picking import role_manual_controls_enabled
from haemolynx.pipeline.checks import check_manual_inlets_outlets_while_masks_choose
from haemolynx.pipeline.stages import assign_boundaries

from test_large_vessel_network_mode import (
    _assign_boundaries_settings,
    _chain_with_large_vessel_masks,
    _network_for_large_vessel_mode,
)


@pytest.mark.parametrize(
    "automated, switch, expected",
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_the_masks_choose_only_with_both_switches_on(automated, switch, expected):
    settings = {"automated_vessel_assignment": automated,
                "inlets_outlets_from_vessel_masks": switch}
    assert inlets_outlets_from_vessel_masks(settings) is expected


def test_a_config_without_the_new_switch_keeps_its_old_behaviour():
    assert inlets_outlets_from_vessel_masks({"automated_vessel_assignment": True}) is True


@pytest.mark.parametrize("role", ["inlet", "outlet"])
def test_the_inlet_and_outlet_tabs_open_when_the_masks_do_not_choose(role):
    masks_choose = {"automated_vessel_assignment": True,
                    "inlets_outlets_from_vessel_masks": True}
    by_hand = {"automated_vessel_assignment": True,
               "inlets_outlets_from_vessel_masks": False}
    assert not role_manual_controls_enabled(role, masks_choose)
    assert role_manual_controls_enabled(role, by_hand)
    assert role_manual_controls_enabled(role, {"automated_vessel_assignment": False})


def test_preflight_warns_when_hand_picked_inlets_would_be_ignored():
    picked = {"automated_vessel_assignment": True,
              "inlets_outlets_from_vessel_masks": True,
              "inlet_node_ids": [3], "outlet_node_volumes": [[[0, 0, 0], [1, 1, 1]]]}
    warnings = check_manual_inlets_outlets_while_masks_choose(picked).warnings
    assert len(warnings) == 1
    assert "inlet_node_ids, outlet_node_volumes" in warnings[0]
    assert "inlets_outlets_from_vessel_masks off" in warnings[0]
    picked["inlets_outlets_from_vessel_masks"] = False
    assert not check_manual_inlets_outlets_while_masks_choose(picked).warnings


def test_with_the_switch_on_the_masks_still_choose(tmp_path):
    G, arteriole, venule = _chain_with_large_vessel_masks()
    settings = _assign_boundaries_settings(plot_dir=tmp_path)

    assign_boundaries(settings, _network_for_large_vessel_mode(G, arteriole, venule, tmp_path))

    assert settings["inlet_nodes"] == [0] and settings["outlet_nodes"] == [9]


def test_with_the_switch_off_the_hand_picked_inlets_are_used_and_the_masks_kept(tmp_path):
    G, arteriole, venule = _chain_with_large_vessel_masks()
    settings = _assign_boundaries_settings(
        plot_dir=tmp_path,
        inlets_outlets_from_vessel_masks=False,
        inlet_node_selection_method="node_ids",
        inlet_node_ids=[3],
        outlet_node_selection_method="node_ids",
        outlet_node_ids=[6],
    )

    boundaries = assign_boundaries(
        settings, _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    )

    assert settings["inlet_nodes"] == [3] and settings["outlet_nodes"] == [6]
    assert boundaries.resistance_node_pair == (3, 6)
    # The masks still drive the large-vessel treatment: its own inlet/outlet
    # come from them, and the edges inside them stay in the network.
    assert settings["large_vessel_inlet_nodes"] == [0]
    assert settings["large_vessel_outlet_nodes"] == [9]
    assert boundaries.graph.number_of_edges() == G.number_of_edges()
    # Nothing the user set is overwritten with the masks' choice.
    assert settings["inlet_node_coordinates"] == []
    assert settings["outlet_node_coordinates"] == []


def test_a_venule_mask_with_no_stump_names_the_tab_to_pick_the_outlet_on(tmp_path):
    """A large venule mask that never reaches the image's edge has no stump:
    the run says where to choose the outlet instead of only that none was found."""
    G, arteriole, venule = _chain_with_large_vessel_masks()
    venule[:] = False
    venule[5:7, 4:6, 4:6] = True        # inside the image, touching no face
    settings = _assign_boundaries_settings(plot_dir=tmp_path)

    with pytest.raises(ValueError, match="Large vessel outlet") as caught:
        assign_boundaries(settings, _network_for_large_vessel_mode(G, arteriole, venule, tmp_path))
    assert "no stump" in str(caught.value)


def test_picking_the_large_vessel_outlet_by_node_id_runs(tmp_path):
    G, arteriole, venule = _chain_with_large_vessel_masks()
    venule[:] = False
    venule[5:7, 4:6, 4:6] = True
    settings = _assign_boundaries_settings(
        plot_dir=tmp_path,
        large_vessel_outlet_node_selection_method="node_ids",
        large_vessel_outlet_node_ids=[9],
    )

    assign_boundaries(settings, _network_for_large_vessel_mode(G, arteriole, venule, tmp_path))

    assert settings["inlet_nodes"] == [0] and settings["outlet_nodes"] == [9]
