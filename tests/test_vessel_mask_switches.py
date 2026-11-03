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


@pytest.mark.parametrize("method, listed", [("node_ids", [999]), ("coordinates", [(5.0, 5.0, 5.0)])])
def test_a_manual_vessel_boundary_list_is_not_read_while_the_small_masks_choose(
    tmp_path, method, listed
):
    """Regression: the panel greys the Arteriole and Venule tabs while the
    small masks choose those boundaries, and a batch does not read them; but
    the stage still picked from their lists first, so node IDs left from an
    earlier build stopped the run, before the masks' boundaries replaced them."""
    names = {"node_ids": "arteriole_boundary_node_ids", "coordinates": "arteriole_boundary_node_coordinates"}
    settings = _small_mask_settings(
        tmp_path,
        automated_vessel_assignment=True,
        arteriole_boundary_selection_method=method,
        **{names[method]: listed},
    )
    expected = _small_mask_settings(tmp_path, automated_vessel_assignment=True)
    assign_boundaries(expected, _network_with_small_masks(tmp_path))

    assign_boundaries(settings, _network_with_small_masks(tmp_path))

    assert settings["arteriole_boundary_nodes"] == expected["arteriole_boundary_nodes"]
    assert settings["venule_boundary_nodes"] == expected["venule_boundary_nodes"]


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


def _mask_file(path, shape, box):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[box] = 255
    tifffile.imwrite(path, mask)
    return path, mask != 0


def _large_mask_run(tmp_path, **overrides):
    """A run's settings with the large masks in use and every load-time
    clean-up off, so a loaded mask is its file."""
    return {
        **SCHEMA.defaults(),
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "large_vessel_mask_dilation_microns": 0.0,
        "large_vessel_min_component_volume_um3": 0.0,
        "large_vessel_swap_minority_components": False,
        "large_vessel_remove_small_opposite_attached_components": False,
        "exclude_smaller_overlapping_volumes": False,
        "plot_dir": tmp_path,
        **overrides,
    }


def _volume(tmp_path, shape=(10, 10, 10), voxel=1.0, downsample=None):
    from haemolynx.pipeline.stages import SkeletonisedVolume

    image = np.zeros(shape, dtype=np.uint8)
    return SkeletonisedVolume(
        image=image,
        skeleton=image.copy(),
        voxel_size_xyz=(voxel,) * 3,
        voxel_size_zyx=(voxel,) * 3,
        output_dir=tmp_path,
        downsample=downsample,
    )


def test_masks_switched_on_in_a_resumed_downsampled_run_are_reduced_onto_its_grid(tmp_path):
    """Regression: the masks loaded for a role ticked since the run it resumes
    were read without the image's reduction, so on a downsampled image the
    run stopped: the mask's whole shape is not the reduced image's."""
    from haemolynx.io import clear_vessel_mask_cache
    from haemolynx.io.downsample import Downsample, block_reduce_volume

    clear_vessel_mask_cache()
    whole = (10, 10, 10)
    arteriole_path, arteriole = _mask_file(tmp_path / "a.tif", whole, np.s_[0:4, 4:8, 4:8])
    venule_path, venule = _mask_file(tmp_path / "v.tif", whole, np.s_[6:10, 4:8, 4:8])
    downsample = Downsample(2, (2, 2, 2), whole, (1.0, 1.0, 1.0))
    network = VesselNetwork(
        graph=_chain_with_large_vessel_masks()[0],
        volume=_volume(tmp_path, shape=downsample.shape_zyx, voxel=2.0, downsample=downsample),
    )
    settings = _large_mask_run(
        tmp_path, large_arteriole_mask_path=arteriole_path, large_venule_mask_path=venule_path
    )

    try:
        now = _vessel_masks_now_in_use(settings, network)
    finally:
        clear_vessel_mask_cache()

    assert np.array_equal(now.large_arteriole_mask, block_reduce_volume(arteriole, (2, 2, 2), "majority"))
    assert np.array_equal(now.large_venule_mask, block_reduce_volume(venule, (2, 2, 2), "majority"))


@pytest.mark.parametrize(
    "change",
    ["another arteriole file", "a venule file entered", "a load-time dilation"],
)
def test_a_resumed_network_loads_its_masks_again_when_what_they_are_read_from_changed(
    tmp_path, change
):
    """Regression: a run resumed on the Boundaries tab kept the masks of the
    run it resumed whenever their role was still in use, so another mask file,
    a load-time clean-up setting changed, or a venule mask entered beside a
    global one went unread -- the last stopping the run for want of a venule
    mask, the others assigning the boundaries from the old masks."""
    from haemolynx.io import clear_vessel_mask_cache
    from haemolynx.pipeline.stages import build_network

    clear_vessel_mask_cache()
    shape = (10, 10, 10)
    first_path, first = _mask_file(tmp_path / "a1.tif", shape, np.s_[0:3, 4:6, 4:6])
    second_path, second = _mask_file(tmp_path / "a2.tif", shape, np.s_[0:5, 3:7, 3:7])
    venule_path, venule = _mask_file(tmp_path / "v.tif", shape, np.s_[7:10, 4:6, 4:6])
    before = _large_mask_run(
        tmp_path,
        large_arteriole_mask_path=first_path,
        large_venule_mask_path=None if change == "a venule file entered" else venule_path,
    )
    after = {
        "another arteriole file": {"large_arteriole_mask_path": second_path},
        "a venule file entered": {"large_venule_mask_path": venule_path},
        "a load-time dilation": {"large_vessel_mask_dilation_microns": 1.0},
    }[change]
    try:
        network = build_network(
            before, _volume(tmp_path), SCHEMA, resumed_graph=_chain_with_large_vessel_masks()[0]
        )
        now = _vessel_masks_now_in_use({**before, **after}, network)
        unchanged = _vessel_masks_now_in_use(before, network)
    finally:
        clear_vessel_mask_cache()

    expected = {
        "another arteriole file": second,
        "a venule file entered": first,
        "a load-time dilation": None,
    }[change]
    if expected is None:
        assert np.count_nonzero(now.large_arteriole_mask) > np.count_nonzero(first)
    else:
        assert np.array_equal(np.asarray(now.large_arteriole_mask) != 0, expected)
    assert now.large_venule_mask is not None
    assert np.array_equal(np.asarray(now.large_venule_mask) != 0, venule) or change == "a load-time dilation"
    # Nothing changed: the masks the network holds are handed on as they are.
    assert unchanged is network


def test_a_synthetic_network_uses_no_vessel_masks_whatever_the_switches_hold():
    """The generator chooses a synthetic network's boundaries, and there is no
    image for a mask to lie on: every mask switch reads as off."""
    from haemolynx.graph import inlets_outlets_from_vessel_masks

    settings = {
        **SCHEMA.defaults(),
        "network_source": "synthetic",
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "use_small_vessel_masks_for_boundary_assignment": True,
        "inlets_outlets_from_vessel_masks": True,
        "assign_large_vessel_branch_orders": True,
        "cut_network_at_large_vessel_volumes": False,
    }
    assert vessel_masks_in_use(settings, "large") is False
    assert vessel_masks_in_use(settings, "small") is False
    assert inlets_outlets_from_vessel_masks(settings) is False
    assert assigns_large_vessel_branch_orders(settings) is False
    image_run = {**settings, "network_source": "segmented_image"}
    assert vessel_masks_in_use(image_run, "large") is True
