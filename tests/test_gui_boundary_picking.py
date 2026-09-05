"""Turning boundary settings into layers and back, without a display.

The two layers this module describes are not views of the settings, they *are*
the settings -- napari world coordinates are already the microns the settings
store. That only holds if the conversions are exact both ways and if nothing
numpy-shaped escapes into a settings row, so both are pinned here.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

from haemolynx.graph.boundaries import BOUNDARY_ROLE_SETTINGS  # noqa: E402
from haemolynx.gui.boundary_picking import (  # noqa: E402
    BC_COORDINATES,
    BC_LAYER_NAMES,
    BC_REGION_NAMES,
    regions_name,
    ROLES,
    BoundaryPicks,
    box_from_rectangle,
    coordinate_setting,
    group_for,
    plain,
    rectangle_from_box,
    settings_from_layers,
    snap,
    specs_for,
    terminal_points,
    volume_setting,
    wanted_rows,
)
from haemolynx.gui.results import LAYER_NAMES, role_colours  # noqa: E402
from haemolynx.pipeline import default_schema  # noqa: E402

A_BOX = ([0.0, 1308.0, 0.0], [600.0, 1730.0, 700.0])

CONFIGURED = {
    "inlet_node_coordinates": [[152.0, 340.0, 527.0], [160.0, 350.0, 545.0]],
    "outlet_node_volumes": [list(A_BOX)],
}


# --- the roles come from the selector, not from a second list ----------------


def test_the_roles_are_the_selectors_own():
    """A fifth role must reach the panel without anything here changing."""
    assert ROLES == tuple(BOUNDARY_ROLE_SETTINGS)


@pytest.mark.parametrize("role", ROLES)
def test_each_role_names_the_settings_the_run_will_read(role):
    assert coordinate_setting(role) == BOUNDARY_ROLE_SETTINGS[role]["coordinates"]
    assert volume_setting(role) == BOUNDARY_ROLE_SETTINGS[role]["volume_boxes"]
    assert coordinate_setting(role) in default_schema()
    assert volume_setting(role) in default_schema()


# --- a box and the rectangle that draws it are exact inverses ----------------


def test_a_box_becomes_a_planar_rectangle_at_its_centre():
    """Shapes cannot draw a 3D box, so the rectangle carries the y/x extent."""
    corners, depth = rectangle_from_box(*A_BOX)

    assert corners.shape == (4, 3)
    assert len(set(corners[:, 0])) == 1, "a rectangle must be planar"
    assert corners[0, 0] == pytest.approx(300.0), "drawn at the box's z centre"
    assert depth == pytest.approx(600.0)
    assert corners[:, 1].min() == pytest.approx(1308.0)
    assert corners[:, 2].max() == pytest.approx(700.0)


def test_the_rectangle_turns_back_into_the_box_it_came_from():
    """The round trip is what lets the layer be treated as the setting."""
    corners, depth = rectangle_from_box(*A_BOX)
    assert box_from_rectangle(corners, depth=depth) == [list(A_BOX[0]), list(A_BOX[1])]


def test_corner_order_does_not_matter():
    """`boundaries.py` normalises lo/hi, so neither end of a drag is special."""
    forwards, depth = rectangle_from_box(*A_BOX)
    backwards, other = rectangle_from_box(A_BOX[1], A_BOX[0])
    assert np.allclose(forwards, backwards)
    assert depth == pytest.approx(other)


def test_a_rectangle_drawn_on_a_slice_grows_both_ways():
    """Drawn at the middle of a stack, a depth means half either side."""
    corners = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 5.0],
                        [10.0, 5.0, 5.0], [10.0, 5.0, 0.0]])
    lo, hi = box_from_rectangle(corners, depth=4.0)
    assert lo[0] == pytest.approx(8.0)
    assert hi[0] == pytest.approx(12.0)


def test_a_depth_of_zero_gives_a_flat_box():
    corners, _ = rectangle_from_box([5.0, 0.0, 0.0], [5.0, 9.0, 9.0])
    lo, hi = box_from_rectangle(corners, depth=0.0)
    assert lo[0] == pytest.approx(hi[0] == pytest.approx(5.0) and 5.0)


def test_a_region_with_too_few_corners_is_refused():
    with pytest.raises(ValueError, match="three"):
        box_from_rectangle(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]), depth=1.0)


# --- reading the settings ----------------------------------------------------


def test_it_reads_every_role_from_a_config():
    picks = BoundaryPicks.from_settings(CONFIGURED)
    assert len(picks.coordinates["inlet"]) == 2
    assert len(picks.volumes["outlet"]) == 1
    assert picks.coordinates["outlet"] == ()
    assert not picks.problems


def test_an_unreadable_entry_is_reported_rather_than_raised():
    """A panel must not fall over on a hand-edited config."""
    picks = BoundaryPicks.from_settings(
        {
            "inlet_node_coordinates": [[1.0, 2.0], [1.0, 2.0, 3.0], "nonsense"],
            "outlet_node_volumes": [[[0.0, 0.0, 0.0]]],
        }
    )
    assert picks.coordinates["inlet"] == ((1.0, 2.0, 3.0),), "the good one survives"
    assert picks.volumes["outlet"] == ()
    assert len(picks.problems) == 3
    assert "inlet_node_coordinates[0]" in picks.problems[0]


def test_a_coordinate_that_is_not_finite_is_not_a_coordinate():
    picks = BoundaryPicks.from_settings(
        {"inlet_node_coordinates": [[float("nan"), 1.0, 2.0]]}
    )
    assert picks.coordinates["inlet"] == ()
    assert picks.problems


def test_reading_nothing_is_not_an_error():
    picks = BoundaryPicks.from_settings({})
    assert all(picks.coordinates[role] == () for role in ROLES)
    assert picks.summary() == "No boundary conditions configured."


# --- the round trip, and the fixed point that makes syncing safe -------------


def test_settings_survive_a_round_trip():
    picks = BoundaryPicks.from_settings(CONFIGURED)
    out = picks.to_settings()
    assert out["inlet_node_coordinates"] == CONFIGURED["inlet_node_coordinates"]
    assert out["outlet_node_volumes"] == CONFIGURED["outlet_node_volumes"]


def test_a_second_round_trip_changes_nothing():
    """The property that stops a settings-layer-settings sync oscillating."""
    once = BoundaryPicks.from_settings(CONFIGURED).to_settings()
    twice = BoundaryPicks.from_settings(once).to_settings()
    assert once == twice


# --- nothing numpy-shaped reaches a settings row -----------------------------


def test_values_written_to_a_row_are_builtin_floats():
    """magicgui stores `str(value)` and reads it back with `literal_eval`.

    `repr(np.float64(1.5))` is 'np.float64(1.5)', which does not parse, so a
    numpy scalar in a row breaks that row the next time anyone touches it.
    """
    picks = BoundaryPicks.from_settings(
        {"inlet_node_coordinates": np.array([[1.5, 2.5, 3.5]])}
    )
    value = picks.to_settings()["inlet_node_coordinates"]
    assert all(type(v) is float for v in value[0])
    assert ast.literal_eval(str(value)) == value


def test_values_written_to_a_row_can_be_saved_as_yaml():
    """`yaml.safe_dump` refuses a np.float64 outright, so saving would fail."""
    settings = BoundaryPicks.from_settings(
        {"outlet_node_volumes": np.array([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])}
    ).to_settings()
    assert yaml.safe_load(yaml.safe_dump(settings)) == settings


def test_values_written_to_a_row_validate_against_the_schema():
    settings = BoundaryPicks.from_settings(CONFIGURED).to_settings()
    default_schema().validate(settings)


def test_plain_reaches_all_the_way_down():
    nested = np.array([[[1.0, 2.0, 3.0]]])
    assert plain(nested) == [[[1.0, 2.0, 3.0]]]
    assert plain(np.float64(2.5)) == 2.5
    assert type(plain(np.float64(2.5))) is float


# --- the layer specs ---------------------------------------------------------


def test_the_coordinates_layer_exists_even_with_nothing_on_it():
    """It is the surface you click into, so it cannot wait for a first point."""
    specs = specs_for({})
    assert [spec.name for spec in specs] == [BC_COORDINATES]
    assert specs[0].data.shape == (0, 3)


def test_the_regions_layer_appears_only_when_there_is_a_region():
    """One layer per role, and only for a role that has something in it: an
    empty Shapes layer draws nothing and is one more row in the layer list."""
    assert [s.name for s in specs_for(CONFIGURED)] == [
        BC_COORDINATES, regions_name("outlet"),
    ]
    assert [s.name for s in specs_for({"inlet_node_coordinates": [[1.0, 2.0, 3.0]]})] == [
        BC_COORDINATES
    ]


def test_both_layers_stay_in_micron_space():
    """No conversion anywhere: world coordinates already are the setting."""
    for spec in specs_for(CONFIGURED):
        assert spec.scale == (1.0, 1.0, 1.0)


def test_points_carry_their_role_in_settings_order():
    values = {
        "outlet_node_coordinates": [[9.0, 9.0, 9.0]],
        "inlet_node_coordinates": [[1.0, 1.0, 1.0]],
    }
    spec = specs_for(values)[0]
    assert list(spec.features["role"]) == ["inlet", "outlet"]
    assert spec.data[0].tolist() == [1.0, 1.0, 1.0]


def test_regions_carry_their_role_and_their_own_depth():
    """Every shape of a region is tagged, handle and outline alike, so the
    colouring covers the whole box and the sync can tell them apart."""
    spec = specs_for(CONFIGURED)[1]
    assert set(spec.features["role"]) == {"outlet"}
    assert list(spec.features["depth"]) == pytest.approx([600.0] * 13)
    handles = [i for i, part in enumerate(spec.features["part"]) if part == "handle"]
    assert len(handles) == 1
    assert spec.options["shape_type"][handles[0]] == "rectangle"


def test_the_picking_layers_never_collide_with_a_runs_layers():
    assert BC_LAYER_NAMES.isdisjoint(LAYER_NAMES)


def test_the_layers_are_coloured_by_the_same_roles_a_run_uses():
    for spec in specs_for(CONFIGURED):
        assert spec.colour_by == "role"
        assert spec.colour_kind == "categorical"
        assert spec.colour_cycle == role_colours()


def test_the_group_says_what_is_configured():
    group = group_for(CONFIGURED)
    assert "2 inlet coordinates" in group.note and "1 outlet region" in group.note


def test_the_group_says_when_something_could_not_be_read():
    group = group_for({"inlet_node_coordinates": ["nonsense"]})
    assert "could not be read" in group.note


# --- layers back to settings -------------------------------------------------


def test_points_become_the_coordinate_settings():
    out = settings_from_layers(
        points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        point_roles=["outlet", "inlet"],
    )
    assert out["inlet_node_coordinates"] == [[4.0, 5.0, 6.0]]
    assert out["outlet_node_coordinates"] == [[1.0, 2.0, 3.0]]
    assert out["arteriole_boundary_node_coordinates"] == []


def test_rectangles_become_the_volume_settings():
    corners, depth = rectangle_from_box(*A_BOX)
    out = settings_from_layers(
        rectangles=[corners], rectangle_roles=["outlet"], depths=[depth]
    )
    assert out["outlet_node_volumes"] == [list(A_BOX)]
    assert out["inlet_node_volumes"] == []


def test_syncing_one_layer_leaves_the_others_settings_alone():
    """Otherwise editing a point would silently clear every region."""
    out = settings_from_layers(points=np.empty((0, 3)), point_roles=[])
    assert "inlet_node_volumes" not in out
    assert out["inlet_node_coordinates"] == []


def test_only_the_settings_that_would_change_are_written():
    """Writing a row fires its `changed`, so writing an unchanged one is noise."""
    current = {"inlet_node_coordinates": [[1.0, 2.0, 3.0]], "outlet_node_volumes": []}
    same = wanted_rows({"inlet_node_coordinates": [[1.0, 2.0, 3.0]]}, current)
    assert same == {}
    different = wanted_rows({"inlet_node_coordinates": [[9.0, 9.0, 9.0]]}, current)
    assert different == {"inlet_node_coordinates": [[9.0, 9.0, 9.0]]}


def test_a_numpy_value_is_not_mistaken_for_a_change():
    current = {"inlet_node_coordinates": [[1.0, 2.0, 3.0]]}
    assert wanted_rows({"inlet_node_coordinates": np.array([[1.0, 2.0, 3.0]])},
                       current) == {}


# --- snapping ----------------------------------------------------------------


def a_graph():
    import networkx as nx

    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.array([0.0, 0.0, 100.0]))
    graph.add_node(2, pos=np.array([0.0, 0.0, 50.0]))
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 2)      # keeps node 2 above degree 1
    return graph


def test_only_degree_one_nodes_are_candidates():
    """The same rule the selector uses, so the panel cannot over-promise."""
    positions, ids = terminal_points(a_graph())
    assert sorted(ids.tolist()) == [0, 1]
    assert positions.shape == (2, 3)


def test_a_graph_that_is_not_there_yields_no_candidates():
    positions, ids = terminal_points(None)
    assert positions.shape == (0, 3) and len(ids) == 0


def test_snapping_moves_each_point_to_its_nearest_candidate():
    positions, _ = terminal_points(a_graph())
    snapped, moved = snap(np.array([[0.0, 0.0, 90.0]]), positions)
    assert snapped[0].tolist() == [0.0, 0.0, 100.0]
    assert moved[0] == pytest.approx(10.0)


def test_snapping_with_nothing_to_snap_to_is_a_no_op():
    """Before a run there is no graph, and that is not an error."""
    points = np.array([[1.0, 2.0, 3.0]])
    snapped, moved = snap(points, np.empty((0, 3)))
    assert snapped.tolist() == points.tolist()
    assert moved.tolist() == [0.0]


def test_a_tie_goes_to_the_lowest_index_every_time():
    candidates = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    snapped, _ = snap(np.array([[0.0, 0.0, 0.0]]), candidates)
    assert snapped[0].tolist() == [0.0, 0.0, -1.0]


# --- the module stays pure ---------------------------------------------------


def test_importing_it_does_not_drag_in_a_gui():
    """It has to be testable without napari, like the rest of the pure half."""
    probe = (
        "import sys, haemolynx.gui.boundary_picking as m; "
        "bad = [n for n in ('napari', 'qtpy', 'magicgui') if n in sys.modules]; "
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert out.stdout.strip() == "[]", out.stdout


# --- a region is drawn as the box it stands for, not just a rectangle --------


def test_a_region_is_drawn_as_a_box_not_a_flat_rectangle():
    """The rectangle is the handle; the twelve segments are the box.

    A rectangle alone is planar, so on its own it says nothing about how deep
    a region is -- which is the whole of what a volume box adds.
    """
    from haemolynx.gui.boundary_picking import region_shapes

    picks = BoundaryPicks.from_settings({"outlet_node_volumes": [list(A_BOX)]})
    _data, kinds, features = region_shapes(picks)

    assert kinds.count("rectangle") == 1, "one editable handle per region"
    assert kinds.count("line") == 12, "the twelve edges of the box"
    assert list(features["part"]).count("handle") == 1
    assert set(features["role"]) == {"outlet"}


def test_the_outline_spans_the_boxs_full_depth():
    from haemolynx.gui.boundary_picking import box_outline

    edges = box_outline(*A_BOX)
    corners = np.concatenate(edges, axis=0)
    assert corners[:, 0].min() == pytest.approx(A_BOX[0][0])
    assert corners[:, 0].max() == pytest.approx(A_BOX[1][0])
    assert corners[:, 2].max() == pytest.approx(A_BOX[1][2])


def test_every_outline_segment_lies_along_one_axis():
    """An edge of a box changes in exactly one coordinate."""
    from haemolynx.gui.boundary_picking import box_outline

    for edge in box_outline(*A_BOX):
        assert int(np.count_nonzero(edge[0] != edge[1])) == 1


def test_a_flat_region_still_draws_its_rectangle():
    """Zero depth is a legitimate box; it just has no z edges to draw."""
    from haemolynx.gui.boundary_picking import region_shapes

    picks = BoundaryPicks.from_settings(
        {"outlet_node_volumes": [[[5.0, 0.0, 0.0], [5.0, 9.0, 9.0]]]}
    )
    _data, kinds, _features = region_shapes(picks)
    assert kinds.count("rectangle") == 1


# --- only the settings a method reads are shown ------------------------------


def test_a_method_shows_only_what_it_reads():
    from haemolynx.gui.boundary_picking import visible_settings

    wanted = visible_settings(
        {
            "inlet_node_selection_method": "coordinates",
            "outlet_node_selection_method": "volume",
            "arteriole_boundary_selection_method": "all_degree_1",
            "venule_boundary_selection_method": "degree_1_from_inlet",
        }
    )
    assert "inlet_node_coordinates" in wanted
    assert "outlet_node_volumes" in wanted
    assert "boundary_distance_from_inlet_node" in wanted
    # `all_degree_1` takes every terminal, so it configures nothing.
    assert "arteriole_boundary_node_coordinates" not in wanted
    assert "inlet_node_volumes" not in wanted
    assert "boundary_axis" not in wanted


def test_the_band_settings_appear_for_any_role_that_uses_them():
    """They are shared, so one role asking for them is enough -- but a role
    only asks for the percentage at its own end of the axis."""
    from haemolynx.gui.boundary_picking import visible_settings

    wanted = visible_settings({"venule_boundary_selection_method": "edge_percent"})
    assert {"boundary_axis", "boundary_last_percent"} <= wanted
    assert "boundary_first_percent" not in wanted

    both = visible_settings({"venule_boundary_selection_method": "edge_percent",
                             "inlet_node_selection_method": "edge_percent"})
    assert {"boundary_axis", "boundary_first_percent", "boundary_last_percent"} <= both


@pytest.mark.parametrize(
    "method", ["coordinates", "volume", "edge_percent", "all_degree_1",
               "degree_1_from_inlet"]
)
def test_every_method_the_schema_allows_is_accounted_for(method):
    """A method with no rule would silently show nothing."""
    from haemolynx.gui.boundary_picking import settings_for_method

    assert method in default_schema()["inlet_node_selection_method"].choices
    for name in settings_for_method("inlet", method):
        assert name in default_schema()


def test_the_ordering_puts_a_method_above_what_it_reads():
    from haemolynx.gui.boundary_picking import orderable_settings

    order = orderable_settings()
    for role in ROLES:
        assert order.index(f"{role}_selection_method"
                           if f"{role}_selection_method" in order
                           else BOUNDARY_ROLE_SETTINGS[role]["method"]) < order.index(
            coordinate_setting(role)
        )


# --- a role's own settings, and what no role owns ----------------------------


@pytest.mark.parametrize("role", ROLES)
def test_a_roles_settings_start_with_the_method_that_chooses_them(role):
    from haemolynx.gui.boundary_picking import role_settings

    mine = role_settings(role)
    assert mine[0] == BOUNDARY_ROLE_SETTINGS[role]["method"]
    assert set(mine) == {
        BOUNDARY_ROLE_SETTINGS[role]["method"],
        coordinate_setting(role),
        volume_setting(role),
        f"{role}_nodes",
    }
    for name in mine:
        assert name in default_schema(), "a page cannot show a row that does not exist"


def test_no_setting_belongs_to_two_roles():
    """A row lives in one page, so an overlap would lose it from the other."""
    from haemolynx.gui.boundary_picking import role_settings

    seen = [name for role in ROLES for name in role_settings(role)]
    assert len(seen) == len(set(seen))


def test_the_band_settings_belong_to_no_role():
    """One axis and one pair of bands describe the whole network."""
    from haemolynx.gui.boundary_picking import role_settings, shared_settings

    owned = {name for role in ROLES for name in role_settings(role)}
    assert set(shared_settings()).isdisjoint(owned)
    assert "boundary_axis" in shared_settings()


def test_between_them_the_pages_place_every_boundary_setting():
    """Anything neither owns falls through to the tab, which is fine -- but it
    must not fall through by accident, so the split is pinned."""
    from haemolynx.gui.boundary_picking import (
        orderable_settings, role_settings, shared_settings,
    )

    placed = {n for role in ROLES for n in role_settings(role)} | set(shared_settings())
    assert set(orderable_settings()) <= placed


@pytest.mark.parametrize(
    "role,title",
    [("inlet", "Inlet"), ("outlet", "Outlet"),
     ("arteriole_boundary", "Arteriole"), ("venule_boundary", "Venule"),
     ("large_vessel_inlet", "Large vessel inlet"),
     ("large_vessel_outlet", "Large vessel outlet")],
)
def test_a_role_reads_as_a_tab_name(role, title):
    from haemolynx.gui.boundary_picking import role_title

    assert role_title(role) == title


# --- automated assignment greys role sub-tabs (logic, no Qt) -----------------


def test_both_autos_off_keeps_every_role_manual_controls_enabled():
    """Excludes LARGE_VESSEL_NETWORK_ROLES: those two have their own gate,
    assign_large_vessel_branch_orders, not either auto flag -- see
    test_large_vessel_network_mode_gates_the_new_roles below."""
    from haemolynx.gui.boundary_picking import (
        LARGE_VESSEL_NETWORK_ROLES,
        ROLES,
        role_manual_controls_enabled,
    )

    values = {
        "automated_vessel_assignment": False,
        "use_small_vessel_masks_for_boundary_assignment": False,
    }
    for role in ROLES:
        if role in LARGE_VESSEL_NETWORK_ROLES:
            continue
        assert role_manual_controls_enabled(role, values) is True, role


def test_large_vessel_network_mode_gates_the_new_roles():
    """Inverted polarity from LARGE_AUTO_ROLES/SMALL_AUTO_ROLES: these two grey
    while the feature is *off*, since there is nothing to override yet."""
    from haemolynx.gui.boundary_picking import (
        LARGE_VESSEL_NETWORK_ROLES,
        role_manual_controls_enabled,
    )

    off = {"assign_large_vessel_branch_orders": False}
    on = {"assign_large_vessel_branch_orders": True}
    for role in LARGE_VESSEL_NETWORK_ROLES:
        assert role_manual_controls_enabled(role, {}) is False, role
        assert role_manual_controls_enabled(role, off) is False, role
        assert role_manual_controls_enabled(role, on) is True, role


def test_large_auto_disables_inlet_and_outlet_only():
    """``automated_vessel_assignment`` is large-vessel inlet/outlet assignment."""
    from haemolynx.gui.boundary_picking import role_manual_controls_enabled

    values = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "use_small_vessel_masks_for_boundary_assignment": False,
    }
    assert role_manual_controls_enabled("inlet", values) is False
    assert role_manual_controls_enabled("outlet", values) is False
    assert role_manual_controls_enabled("arteriole_boundary", values) is True
    assert role_manual_controls_enabled("venule_boundary", values) is True


def test_small_auto_disables_arteriole_and_venule():
    """Small-vessel masks grey A/V; the root automated toggle also greys I/O."""
    from haemolynx.gui.boundary_picking import role_manual_controls_enabled

    values = {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
    }
    assert role_manual_controls_enabled("inlet", values) is False
    assert role_manual_controls_enabled("outlet", values) is False
    assert role_manual_controls_enabled("arteriole_boundary", values) is False
    assert role_manual_controls_enabled("venule_boundary", values) is False


def test_small_auto_flag_alone_greys_only_arteriole_and_venule():
    """The small-mask setting is what greys A/V, independent of the large gate."""
    from haemolynx.gui.boundary_picking import role_manual_controls_enabled

    values = {
        "automated_vessel_assignment": False,
        "use_small_vessel_masks_for_boundary_assignment": True,
    }
    assert role_manual_controls_enabled("inlet", values) is True
    assert role_manual_controls_enabled("outlet", values) is True
    assert role_manual_controls_enabled("arteriole_boundary", values) is False
    assert role_manual_controls_enabled("venule_boundary", values) is False


# --- coordinates that fall outside the image ---------------------------------


def test_a_coordinate_outside_the_image_is_called_out():
    """Microns mistaken for voxel indices is the one error that self-conceals.

    A run snaps every coordinate to its nearest terminal, so a wrong one does
    not fail -- it quietly selects the wrong vessel.
    """
    from haemolynx.gui.boundary_picking import outside_extent

    notes = outside_extent(
        {"inlet_node_coordinates": [[10.0, 10.0, 10.0], [900.0, 10.0, 10.0]]},
        [0.0, 0.0, 0.0],
        [509.0, 1624.0, 1098.0],
    )
    assert notes == ("1 of 2 inlet coordinate(s)",)


def test_coordinates_inside_the_image_say_nothing():
    from haemolynx.gui.boundary_picking import outside_extent

    assert outside_extent(
        {"inlet_node_coordinates": [[10.0, 10.0, 10.0]]},
        [0.0, 0.0, 0.0], [509.0, 1624.0, 1098.0],
    ) == ()


def test_a_negative_coordinate_is_outside_too():
    from haemolynx.gui.boundary_picking import outside_extent

    assert outside_extent(
        {"outlet_node_coordinates": [[-1.0, 10.0, 10.0]]},
        [0.0, 0.0, 0.0], [509.0, 1624.0, 1098.0],
    ) == ("1 of 1 outlet coordinate(s)",)


def test_the_summary_names_what_it_counted():
    picks = BoundaryPicks.from_settings(CONFIGURED)
    assert picks.summary() == "2 inlet coordinates, 1 outlet region"


# --- which band percentage a role actually reads -----------------------------


def test_an_inlet_role_reads_the_first_percentage():
    from haemolynx.gui.boundary_picking import settings_for_method

    assert settings_for_method("inlet", "edge_percent") == (
        "boundary_axis", "boundary_first_percent",
    )
    assert settings_for_method("arteriole_boundary", "edge_percent") == (
        "boundary_axis", "boundary_first_percent",
    )


def test_an_outlet_role_reads_the_last_percentage():
    """A run computes both ends every time; a role takes only its own."""
    from haemolynx.gui.boundary_picking import settings_for_method

    assert settings_for_method("outlet", "edge_percent") == (
        "boundary_axis", "boundary_last_percent",
    )
    assert settings_for_method("venule_boundary", "edge_percent") == (
        "boundary_axis", "boundary_last_percent",
    )


def test_the_percentage_split_matches_the_selector():
    """`select_boundary_nodes_by_method` gives `edge_percent` to the input end
    and `end_percent` to the output end; the panel must agree or it shows a
    number that changes nothing."""
    from haemolynx.gui.boundary_picking import PERCENT_FOR_NODE_ROLE

    assert set(PERCENT_FOR_NODE_ROLE) == {"inlet", "outlet"}
    for role, names in BOUNDARY_ROLE_SETTINGS.items():
        assert names["node_role"] in PERCENT_FOR_NODE_ROLE


# --- the band an edge_percent role selects from ------------------------------

FULL_IMAGE = ([0.0, 0.0, 0.0], [500.0, 1000.0, 800.0])


def test_an_inlet_band_starts_at_the_low_end_of_the_axis():
    """`edge_percent` is the one method whose region is implied rather than
    written down, so the only way to see what it takes was to run it."""
    from haemolynx.gui.boundary_picking import band_boxes

    boxes = band_boxes(
        {"boundary_axis": 1, "inlet_node_selection_method": "edge_percent",
         "boundary_first_percent": 10.0},
        *FULL_IMAGE,
    )
    lo, hi = boxes["inlet"]
    assert lo[1] == pytest.approx(0.0)
    assert hi[1] == pytest.approx(100.0)


def test_an_outlet_band_ends_at_the_high_end_of_the_axis():
    from haemolynx.gui.boundary_picking import band_boxes

    boxes = band_boxes(
        {"boundary_axis": 1, "outlet_node_selection_method": "edge_percent",
         "boundary_last_percent": 25.0},
        *FULL_IMAGE,
    )
    lo, hi = boxes["outlet"]
    assert lo[1] == pytest.approx(750.0)
    assert hi[1] == pytest.approx(1000.0)


def test_a_band_spans_everything_across_the_other_axes():
    """The selector's rule is on one coordinate alone."""
    from haemolynx.gui.boundary_picking import band_boxes

    lo, hi = band_boxes(
        {"boundary_axis": 1, "inlet_node_selection_method": "edge_percent",
         "boundary_first_percent": 10.0},
        *FULL_IMAGE,
    )["inlet"]
    assert (lo[0], hi[0]) == pytest.approx((0.0, 500.0))
    assert (lo[2], hi[2]) == pytest.approx((0.0, 800.0))


def test_a_band_is_measured_across_the_terminals_when_there_are_any():
    """What a run measures. A network rarely reaches its image's edge, so the
    image-relative band is a different box -- see `select_boundary_terminal_nodes`."""
    from haemolynx.gui.boundary_picking import band_boxes

    lo, hi = band_boxes(
        {"boundary_axis": 1, "inlet_node_selection_method": "edge_percent",
         "boundary_first_percent": 50.0},
        *FULL_IMAGE,
        axis_span=(200.0, 400.0),
    )["inlet"]
    assert (lo[1], hi[1]) == pytest.approx((200.0, 300.0))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_band_follows_the_axis_it_is_told(axis):
    from haemolynx.gui.boundary_picking import band_boxes

    lo, hi = band_boxes(
        {"boundary_axis": axis, "inlet_node_selection_method": "edge_percent",
         "boundary_first_percent": 50.0},
        *FULL_IMAGE,
    )["inlet"]
    assert hi[axis] == pytest.approx(FULL_IMAGE[1][axis] / 2)
    for other in {0, 1, 2} - {axis}:
        assert hi[other] == pytest.approx(FULL_IMAGE[1][other])


def test_only_a_role_using_edge_percent_gets_a_band():
    from haemolynx.gui.boundary_picking import band_boxes

    boxes = band_boxes(
        {"boundary_axis": 1,
         "inlet_node_selection_method": "edge_percent",
         "outlet_node_selection_method": "coordinates",
         "boundary_first_percent": 10.0, "boundary_last_percent": 10.0},
        *FULL_IMAGE,
    )
    assert set(boxes) == {"inlet"}


def test_a_hundred_percent_band_is_the_whole_span():
    from haemolynx.gui.boundary_picking import band_boxes

    lo, hi = band_boxes(
        {"boundary_axis": 1, "inlet_node_selection_method": "edge_percent",
         "boundary_first_percent": 100.0},
        *FULL_IMAGE,
    )["inlet"]
    assert (lo[1], hi[1]) == pytest.approx((0.0, 1000.0))


def test_a_nonsense_axis_draws_nothing_rather_than_raising():
    from haemolynx.gui.boundary_picking import band_boxes

    for axis in (7, -1, None, "y"):
        assert band_boxes(
            {"boundary_axis": axis, "inlet_node_selection_method": "edge_percent",
             "boundary_first_percent": 10.0},
            *FULL_IMAGE,
        ) == {}


def test_a_band_is_drawn_as_a_box_with_no_handle():
    """It is not a region anyone typed, so there is nothing to drag and
    nothing for the settings to read back out of it."""
    from haemolynx.gui.boundary_picking import BAND, band_boxes, region_shapes

    values = {"boundary_axis": 1, "inlet_node_selection_method": "edge_percent",
              "boundary_first_percent": 10.0}
    bands = band_boxes(values, *FULL_IMAGE)
    _data, kinds, features = region_shapes(BoundaryPicks.from_settings(values), bands)

    assert kinds == ["line"] * 12
    assert set(features["part"]) == {BAND}
    assert set(features["role"]) == {"inlet"}


def test_a_band_and_a_configured_region_can_be_drawn_together():
    from haemolynx.gui.boundary_picking import band_boxes, region_shapes

    values = {"boundary_axis": 1, "inlet_node_selection_method": "edge_percent",
              "boundary_first_percent": 10.0, "outlet_node_volumes": [list(A_BOX)]}
    bands = band_boxes(values, *FULL_IMAGE)
    _data, kinds, features = region_shapes(BoundaryPicks.from_settings(values), bands)

    assert kinds.count("rectangle") == 1, "only the configured region is editable"
    assert kinds.count("line") == 24
    assert list(features["part"]).count("handle") == 1


def test_the_span_of_the_terminals_needs_a_graph():
    from haemolynx.gui.boundary_picking import terminal_axis_span

    assert terminal_axis_span(None, 1) is None


def test_the_span_of_the_terminals_is_where_they_reach():
    import networkx as nx
    from haemolynx.gui.boundary_picking import terminal_axis_span

    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 40.0, 0.0]))
    graph.add_node(1, pos=np.array([0.0, 160.0, 0.0]))
    graph.add_edge(0, 1)

    assert terminal_axis_span(graph, 1) == pytest.approx((40.0, 160.0))


# --- green in, red out -------------------------------------------------------


def test_an_inlet_is_green_and_an_outlet_is_red():
    """Where flow enters and where it leaves is the thing being looked for."""
    colours = dict(role_colours())

    assert colours["inlet"][1] > 0.5, "green channel dominant"
    assert colours["inlet"][0] < 0.3 and colours["inlet"][2] < 0.3
    assert colours["outlet"][0] > 0.5, "red channel dominant"
    assert colours["outlet"][1] < 0.3 and colours["outlet"][2] < 0.3


def test_no_two_roles_share_a_colour():
    colours = dict(role_colours())

    assert len(colours) == len(ROLES)
    assert len({tuple(v) for v in colours.values()}) == len(ROLES)


def test_both_picking_layers_use_the_same_colours():
    """A coordinate and the region around it must not disagree about the role."""
    specs = specs_for(CONFIGURED)

    assert len(specs) == 2
    for spec in specs:
        assert spec.colour_by == "role"
        assert spec.colour_cycle == role_colours()


# --- one layer per role ------------------------------------------------------


def test_each_role_draws_into_its_own_layer():
    """A layer carries one colour and one visibility, so sharing one meant
    inlets and outlets could not be told apart or hidden separately."""
    names = [spec.name for spec in specs_for({
        "inlet_node_volumes": [list(A_BOX)],
        "outlet_node_volumes": [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
    })]

    assert names == [BC_COORDINATES, regions_name("inlet"), regions_name("outlet")]


def test_a_region_layer_holds_only_its_own_roles_shapes():
    specs = {spec.name: spec for spec in specs_for({
        "inlet_node_volumes": [list(A_BOX)],
        "outlet_node_volumes": [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
    })}

    for role in ("inlet", "outlet"):
        assert set(specs[regions_name(role)].features["role"]) == {role}
        assert len(specs[regions_name(role)].data) == 13


def test_a_band_goes_in_its_roles_layer_too():
    from haemolynx.gui.boundary_picking import band_boxes

    values = {"boundary_axis": 1, "outlet_node_selection_method": "edge_percent",
              "boundary_last_percent": 10.0}
    names = [spec.name
             for spec in specs_for(values, band_boxes(values, [0, 0, 0], [9.0, 9.0, 9.0]))]

    assert names == [BC_COORDINATES, regions_name("outlet")]


def test_the_layer_names_say_which_role_they_are():
    assert regions_name("inlet") == "HaemoLynx BC inlet regions"
    assert regions_name("arteriole_boundary") == "HaemoLynx BC arteriole regions"
    assert len(set(BC_REGION_NAMES)) == len(ROLES)


def test_every_picking_layer_is_named_in_one_place():
    """`_clear_our_layers` and the "is this ours?" check both read this."""
    from haemolynx.gui.boundary_picking import BC_LAYER_NAMES

    assert BC_LAYER_NAMES == {BC_COORDINATES, *BC_REGION_NAMES}
    assert BC_LAYER_NAMES.isdisjoint(LAYER_NAMES), "never a run's own layer"
