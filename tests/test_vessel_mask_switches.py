"""A vessel-mask switch left ticked under a switch the user unticked is off.

The Boundaries tab hides every row under "Automated vessel assignment", "Use
large vessel masks" and "Use small vessel masks for boundary assignment" while
that switch is off -- but a hidden row keeps its value. Read raw, those values
carried on: unticking "Use large vessel masks" left the Inlet and Outlet tabs
greyed and the run still taking its inlets from masks (a resumed run kept the
masks of the run it resumed), a small-mask switch left on under an unticked
automated assignment still assigned the arteriole/venule boundaries, and
preflight failed the run on switches the user could no longer see. Every
reader now asks ``graph.vessel_masks_in_use`` and
``graph.assigns_large_vessel_branch_orders`` instead.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import tifffile

from haemolynx.graph import assigns_large_vessel_branch_orders, vessel_masks_in_use
from haemolynx.gui.boundary_picking import role_manual_controls_enabled
from haemolynx.gui.form import shared_ilastik_host
from haemolynx.parsers import check_settings
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.checks import (
    check_ilastik_executable,
    check_large_vessel_branch_order_mode_prerequisites,
)
from haemolynx.pipeline.stages import (
    VesselNetwork,
    _vessel_masks_now_in_use,
    assign_boundaries,
)

from test_large_vessel_network_mode import (
    _assign_boundaries_settings,
    _chain_with_large_vessel_masks,
    _network_for_large_vessel_mode,
)

SCHEMA = default_schema()
SWITCH = {
    "large": "use_large_vessel_masks",
    "small": "use_small_vessel_masks_for_boundary_assignment",
}
#: An arteriole and a venule mask for each role, never read: the networks
#: here carry their masks. Without a venule mask a role's mask is one global
#: mask of every vessel, which assigns no inlets or branch orders.
MASK_PATHS = {
    "large_arteriole_mask_path": "large_a.tif",
    "large_venule_mask_path": "large_v.tif",
    "small_arteriole_mask_path": "small_a.tif",
    "small_venule_mask_path": "small_v.tif",
}


# --- the rule ----------------------------------------------------------------


@pytest.mark.parametrize("role", ["large", "small"])
@pytest.mark.parametrize(
    "automated, switch, expected",
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_masks_are_in_use_only_with_the_gate_and_their_own_switch(
    role, automated, switch, expected
):
    settings = {"automated_vessel_assignment": automated, SWITCH[role]: switch}
    assert vessel_masks_in_use(settings, role) is expected


def test_an_unknown_mask_role_is_rejected():
    with pytest.raises(ValueError, match="'large' or 'small'"):
        vessel_masks_in_use({}, "medium")


@pytest.mark.parametrize(
    "change, expected",
    [
        ({}, True),
        ({"use_large_vessel_masks": False}, False),
        ({"automated_vessel_assignment": False}, False),
        ({"cut_network_at_large_vessel_volumes": True}, False),
        ({"assign_large_vessel_branch_orders": False}, False),
    ],
)
def test_the_large_vessel_tier_needs_the_large_masks_and_no_cut(change, expected):
    """Each of these hides the switch in the panel, so each turns it off."""
    on = {
        **MASK_PATHS,
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "cut_network_at_large_vessel_volumes": False,
        "assign_large_vessel_branch_orders": True,
    }
    assert assigns_large_vessel_branch_orders({**on, **change}) is expected


# --- the panel ---------------------------------------------------------------


def test_unticking_the_large_masks_opens_the_inlet_outlet_and_closes_the_large_tabs():
    left_on = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": False,
        "inlets_outlets_from_vessel_masks": True,
        "cut_network_at_large_vessel_volumes": False,
        "assign_large_vessel_branch_orders": True,
    }
    assert role_manual_controls_enabled("inlet", left_on)
    assert role_manual_controls_enabled("outlet", left_on)
    assert not role_manual_controls_enabled("large_vessel_inlet", left_on)
    assert not role_manual_controls_enabled("large_vessel_outlet", left_on)


def test_a_small_switch_left_on_under_unticked_automated_assignment_greys_nothing():
    left_on = {
        "automated_vessel_assignment": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
    }
    assert role_manual_controls_enabled("arteriole_boundary", left_on)
    assert role_manual_controls_enabled("venule_boundary", left_on)
    in_use = {**left_on, "automated_vessel_assignment": True}
    assert not role_manual_controls_enabled("arteriole_boundary", in_use)


@pytest.mark.parametrize(
    "flag, role",
    [
        ("use_ilastik_large_vessel_segmentation", "large"),
        ("use_ilastik_small_vessel_segmentation", "small"),
    ],
)
def test_the_shared_ilastik_rows_leave_boundaries_with_the_masks(flag, role):
    in_use = {flag: True, "automated_vessel_assignment": True, SWITCH[role]: True}
    assert shared_ilastik_host(in_use) == "boundaries"
    assert shared_ilastik_host({**in_use, SWITCH[role]: False}) is None
    assert shared_ilastik_host({**in_use, "automated_vessel_assignment": False}) is None


# --- preflight ---------------------------------------------------------------


def _errors_naming(report, name):
    return [message for message in report.errors if name in message]


def test_preflight_asks_for_no_large_mask_file_under_unticked_automated_assignment():
    """The paths' own `requires` named only use_large_vessel_masks, so with
    automated assignment unticked over a ticked (hidden) mask switch,
    preflight demanded the default mask files."""
    left_on = {
        **SCHEMA.defaults(),
        "automated_vessel_assignment": False,
        "use_large_vessel_masks": True,
        "large_arteriole_mask_path": "no/such/large_arteriole_mask.tif",
    }
    assert not _errors_naming(check_settings(SCHEMA, left_on), "large_arteriole_mask_path")
    in_use = {**left_on, "automated_vessel_assignment": True}
    assert _errors_naming(check_settings(SCHEMA, in_use), "large_arteriole_mask_path")


@pytest.mark.parametrize(
    "flag, role",
    [
        ("use_ilastik_large_vessel_segmentation", "large"),
        ("use_ilastik_small_vessel_segmentation", "small"),
    ],
)
def test_preflight_wants_no_ilastik_for_a_mask_switch_left_on_hidden(flag, role):
    left_on = {
        flag: True,
        "automated_vessel_assignment": True,
        SWITCH[role]: False,
        "ilastik_executable": None,
    }
    assert not check_ilastik_executable(left_on).errors
    assert check_ilastik_executable({**left_on, SWITCH[role]: True}).errors


def test_preflight_passes_branch_orders_left_on_under_unticked_large_masks():
    left_on = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": False,
        "assign_large_vessel_branch_orders": True,
        "cut_network_at_large_vessel_volumes": False,
    }
    report = check_large_vessel_branch_order_mode_prerequisites(left_on)
    assert not report.errors and not report.warnings


# --- the run -----------------------------------------------------------------


def test_with_the_large_masks_unticked_the_inlets_come_from_the_inlet_tab(tmp_path):
    """The network still carries large masks (a run resumed on the Boundaries
    tab holds those of the run it resumes) and the large-vessel switches under
    "Use large vessel masks" are still ticked, hidden."""
    G, arteriole, venule = _chain_with_large_vessel_masks()
    edges = G.number_of_edges()
    settings = _assign_boundaries_settings(
        **MASK_PATHS,
        plot_dir=tmp_path,
        use_large_vessel_masks=False,
        inlet_node_selection_method="node_ids",
        inlet_node_ids=[3],
        outlet_node_selection_method="node_ids",
        outlet_node_ids=[6],
    )

    boundaries = assign_boundaries(
        settings, _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    )

    assert settings["inlet_nodes"] == [3] and settings["outlet_nodes"] == [6]
    assert settings["large_vessel_inlet_nodes"] == []
    assert settings["large_arteriole_boundary_nodes"] == []
    assert boundaries.graph.number_of_edges() == edges
    assert boundaries.large_arteriole_mask is None and boundaries.large_venule_mask is None

    # Ticked, the same masks choose: they were there to be used.
    G, arteriole, venule = _chain_with_large_vessel_masks()
    settings = _assign_boundaries_settings(**MASK_PATHS, plot_dir=tmp_path)
    assign_boundaries(settings, _network_for_large_vessel_mode(G, arteriole, venule, tmp_path))
    assert settings["inlet_nodes"] == [0]


def _small_mask_settings(tmp_path, **overrides):
    return _assign_boundaries_settings(
        **MASK_PATHS,
        plot_dir=tmp_path,
        use_large_vessel_masks=False,
        assign_large_vessel_branch_orders=False,
        use_small_vessel_masks_for_boundary_assignment=True,
        small_vessel_boundary_assignment_enable_overlap_cleanup=False,
        small_vessel_boundary_fallback_to_hop_distance=False,
        inlet_node_selection_method="node_ids",
        inlet_node_ids=[0],
        outlet_node_selection_method="node_ids",
        outlet_node_ids=[9],
        **overrides,
    )


def _network_with_small_masks(tmp_path):
    G, arteriole, venule = _chain_with_large_vessel_masks()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    return VesselNetwork(
        graph=network.graph,
        volume=network.volume,
        small_arteriole_mask=arteriole,
        small_venule_mask=venule,
    )


def test_small_masks_left_on_under_unticked_automated_assignment_are_not_used(tmp_path):
    settings = _small_mask_settings(tmp_path, automated_vessel_assignment=False)

    boundaries = assign_boundaries(settings, _network_with_small_masks(tmp_path))

    assert settings["arteriole_boundary_nodes"] == []
    assert settings["venule_boundary_nodes"] == []
    assert boundaries.small_arteriole_mask is None


def test_the_small_masks_alone_assign_the_boundaries(tmp_path):
    """Automated assignment with only the small masks ticked: the large
    block used to run regardless, and stopped the run for want of large
    masks."""
    settings = _small_mask_settings(tmp_path, automated_vessel_assignment=True)

    boundaries = assign_boundaries(settings, _network_with_small_masks(tmp_path))

    assert settings["inlet_nodes"] == [0] and settings["outlet_nodes"] == [9]
    assert settings["arteriole_boundary_nodes"] and settings["venule_boundary_nodes"]
    assert boundaries.small_arteriole_mask is not None
    assert boundaries.large_arteriole_mask is None


# --- a run resumed on the Boundaries tab ------------------------------------


def test_a_resumed_network_drops_the_masks_of_a_role_switched_off(tmp_path):
    G, arteriole, venule = _chain_with_large_vessel_masks()
    network = _network_for_large_vessel_mode(G, arteriole, venule, tmp_path)
    G.graph["large_arteriole_mask_voxel_size_xyz"] = (1.0, 1.0, 1.0)
    G.graph["large_venule_mask_voxel_size_xyz"] = (1.0, 1.0, 1.0)
    settings = {"automated_vessel_assignment": True, "use_large_vessel_masks": False}

    now = _vessel_masks_now_in_use(settings, network)

    assert now.large_arteriole_mask is None and now.large_venule_mask is None
    assert "large_arteriole_mask_voxel_size_xyz" not in now.graph.graph
    # Still in use: the network is handed on as it is.
    in_use = {**settings, "use_large_vessel_masks": True}
    assert _vessel_masks_now_in_use(in_use, network) is network


def test_a_resumed_network_loads_the_masks_of_a_role_switched_on(tmp_path):
    G, arteriole, venule = _chain_with_large_vessel_masks()
    network = replace(
        _network_for_large_vessel_mode(G, arteriole, venule, tmp_path),
        large_arteriole_mask=None,
        large_venule_mask=None,
    )
    shape = network.volume.image.shape
    arteriole = np.zeros(shape, dtype=np.uint8)
    arteriole[0:5, 2:7, 2:7] = 255
    venule = np.zeros(shape, dtype=np.uint8)
    venule[5:10, 2:7, 2:7] = 255
    tifffile.imwrite(tmp_path / "small_a.tif", arteriole)
    tifffile.imwrite(tmp_path / "small_v.tif", venule)
    settings = {
        **SCHEMA.defaults(),
        "automated_vessel_assignment": True,
        "use_small_vessel_masks_for_boundary_assignment": True,
        "small_arteriole_mask_path": tmp_path / "small_a.tif",
        "small_venule_mask_path": tmp_path / "small_v.tif",
    }

    now = _vessel_masks_now_in_use(settings, network)

    np.testing.assert_array_equal(np.asarray(now.small_arteriole_mask) != 0, arteriole != 0)
    np.testing.assert_array_equal(np.asarray(now.small_venule_mask) != 0, venule != 0)
    assert now.large_arteriole_mask is None
