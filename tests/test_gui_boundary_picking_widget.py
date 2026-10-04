"""Picking boundary conditions in a real viewer.

`test_gui_boundary_picking.py` decides what the layers should contain without a
display. What is left for here is the part only a real viewer answers: that
editing a layer reaches the settings row in a form the row can read back, that
the two directions do not fight, and that a picked coordinate lands where the
image says it should.
"""
from __future__ import annotations

import ast

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import _clear_our_layers, settings_widget  # noqa: E402
from haemolynx.gui.boundary_picking import (  # noqa: E402
    BC_BOX_NAMES,
    BC_COORDINATES,
    BC_REGION_NAMES,
    ROLES,
    boxes_name,
    regions_name,
    method_setting,
    rectangle_from_box,
)
from haemolynx.gui.results import role_colours  # noqa: E402

pytestmark = pytest.mark.gui

#: A stack with an anisotropic voxel, so a coordinate that was quietly read as
#: a voxel index instead of microns would show up.
VOXEL_ZYX = (2.0, 1.167, 1.167)
A_BOX = [[0.0, 100.0, 0.0], [100.0, 180.0, 150.0]]


@pytest.fixture
def panel(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((60, 200, 200)), name="stack", scale=VOXEL_ZYX)
    widget = settings_widget(napari_viewer=viewer)
    return widget, viewer, widget._haemolynx_boundaries


def rows_of(widget):
    return widget._haemolynx_rows()


def no_bands(widget):
    """Take every role off `edge_percent`.

    Three of the four roles default to it, and each one draws the band it
    would select from -- so a test about a region someone configured has to
    say it is not also asking for three implied ones.
    """
    for role in ROLES:
        rows_of(widget)[method_setting(role)].value = "coordinates"


def drawn_for(viewer, role):
    """Every box vertex drawn for one role, from that role's own box layer."""
    name = boxes_name(role)
    if name not in viewer.layers:
        return np.empty((0, 3))
    return np.asarray(viewer.layers[name].data[0])


def drag(layer, start, end, *, steps=4, view=None, up=None, displayed=(1, 2)):
    """A press, a few moves and a release, through napari's own dispatch.

    The same path a real mouse takes (`mouse_press_callbacks` and friends run
    every callback on the layer, napari's drawing tool included), which is
    what the crash lived in: two callbacks answering one drag.
    """
    from napari.utils._test_utils import read_only_mouse_event
    from napari.utils.interactions import (
        mouse_move_callbacks, mouse_press_callbacks, mouse_release_callbacks,
    )

    common = {"button": 1, "dims_displayed": list(displayed),
              "view_direction": view, "up_direction": up}
    mouse_press_callbacks(layer, read_only_mouse_event(
        type="mouse_press", position=tuple(start), **common))
    for t in np.linspace(0.0, 1.0, steps)[1:]:
        point = np.asarray(start, dtype=float) + t * (
            np.asarray(end, dtype=float) - np.asarray(start, dtype=float))
        mouse_move_callbacks(layer, read_only_mouse_event(
            type="mouse_move", is_dragging=True, position=tuple(point), **common))
    mouse_release_callbacks(layer, read_only_mouse_event(
        type="mouse_release", position=tuple(end), **common))


# --- what a config describes, drawn ------------------------------------------


def test_showing_a_config_puts_both_layers_in_the_viewer(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[10.0, 20.0, 30.0]]
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]

    bc.show()

    assert isinstance(viewer.layers[BC_COORDINATES], napari.layers.Points)
    assert isinstance(viewer.layers[regions_name("outlet")], napari.layers.Shapes)
    assert tuple(viewer.layers[BC_COORDINATES].scale) == (1.0, 1.0, 1.0)


def test_a_picked_coordinate_lands_where_the_image_says_it_should(panel):
    """The claim the whole feature rests on: world coordinates are microns.

    The stack is 60 slices at 2 um, so its far z face is at 118 um. A
    coordinate there must sit at the far face of the image layer, not at slice
    118 -- which is off the end of a 60-slice stack.
    """
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[118.0, 0.0, 0.0]]

    bc.show()

    image_far_z = float(viewer.layers["stack"].extent.world[1][0])
    assert viewer.layers[BC_COORDINATES].data[0][0] == pytest.approx(image_far_z)


def test_the_region_is_drawn_as_a_rectangle_at_the_boxs_centre(panel):
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]

    bc.show()

    corners = viewer.layers[regions_name("outlet")].data[0]
    assert corners.shape == (4, 3)
    assert corners[:, 0].tolist() == [50.0] * 4, "planar, at the box's z centre"
    assert corners[:, 1].min() == pytest.approx(100.0)
    assert corners[:, 2].max() == pytest.approx(150.0)


def test_showing_twice_updates_rather_than_duplicates(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[1.0, 2.0, 3.0]]
    bc.show()
    same = viewer.layers[BC_COORDINATES]

    rows_of(widget)["inlet_node_coordinates"].value = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    bc.show()

    assert viewer.layers[BC_COORDINATES] is same
    assert len(viewer.layers[BC_COORDINATES].data) == 2
    assert sum(l.name == BC_COORDINATES for l in viewer.layers) == 1


def test_a_layer_of_someone_elses_with_that_name_is_not_overwritten(panel):
    widget, viewer, bc = panel
    theirs = viewer.add_points(np.zeros((1, 3)), name=BC_COORDINATES)

    bc.show()

    assert viewer.layers[BC_COORDINATES] is theirs
    assert f"{BC_COORDINATES} (HaemoLynx)" in viewer.layers


def test_clear_layers_takes_the_picking_layers_with_it(panel):
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    bc.show()

    _clear_our_layers(viewer)

    assert BC_COORDINATES not in viewer.layers
    assert all(name not in viewer.layers for name in BC_REGION_NAMES)
    assert all(name not in viewer.layers for name in BC_BOX_NAMES)


# --- the colours, which were wrong before ------------------------------------


def test_each_role_gets_its_own_colour_whatever_order_they_appear_in(panel):
    """A layer holding only outlet nodes used to draw them in starting's blue.

    The cycle was handed to napari as a bare list of colours, which it pairs
    with the values in the order it first encounters them, not by the labels
    they were declared against.
    """
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_coordinates"].value = [[1.0, 1.0, 1.0]]
    rows_of(widget)["inlet_node_coordinates"].value = [[2.0, 2.0, 2.0]]

    bc.show()

    expected = dict(role_colours())
    layer = viewer.layers[BC_COORDINATES]
    for role, colour in zip(layer.features["role"], layer.face_color):
        assert np.allclose(colour, expected[role], atol=0.01), role


def test_the_colours_survive_a_second_show(panel):
    """`options` are applied only on first add, so colour has to come through
    the colour path or it goes stale the moment a layer is updated."""
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[1.0, 1.0, 1.0]]
    bc.show()
    rows_of(widget)["outlet_node_coordinates"].value = [[2.0, 2.0, 2.0]]

    bc.show()

    expected = dict(role_colours())
    layer = viewer.layers[BC_COORDINATES]
    for role, colour in zip(layer.features["role"], layer.face_color):
        assert np.allclose(colour, expected[role], atol=0.01), role


# --- editing a layer edits the settings --------------------------------------


def test_adding_a_point_writes_it_into_the_chosen_roles_setting(panel):
    widget, viewer, bc = panel
    bc.role.value = "outlet"
    bc.pick()

    layer = viewer.layers[BC_COORDINATES]
    layer.data = np.array([[11.0, 22.0, 33.0]])

    assert widget._haemolynx_values()["outlet_node_coordinates"] == [[11.0, 22.0, 33.0]]
    assert widget._haemolynx_values()["inlet_node_coordinates"] == []


def test_the_row_can_read_back_what_was_written_into_it(panel):
    """magicgui stores `str(value)` and parses it with `literal_eval`, so a
    numpy float reaching the row would break it the next time it is touched."""
    widget, viewer, bc = panel
    bc.pick()
    viewer.layers[BC_COORDINATES].data = np.array([[1.5, 2.5, 3.5]])

    value = rows_of(widget)["inlet_node_coordinates"].value
    assert all(type(v) is float for v in value[0]), "a numpy float breaks the row"
    assert ast.literal_eval(str(value)) == [[1.5, 2.5, 3.5]]


def test_deleting_a_point_removes_exactly_that_coordinate(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [
        [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]
    ]
    bc.show()

    layer = viewer.layers[BC_COORDINATES]
    layer.selected_data = {1}
    layer.remove_selected()

    assert widget._haemolynx_values()["inlet_node_coordinates"] == [
        [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]
    ]


def test_moving_a_point_updates_it_in_place(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[1.0, 1.0, 1.0]]
    bc.show()

    layer = viewer.layers[BC_COORDINATES]
    layer.data = np.array([[7.0, 8.0, 9.0]])

    assert widget._haemolynx_values()["inlet_node_coordinates"] == [[7.0, 8.0, 9.0]]


def test_syncing_the_same_edit_twice_changes_nothing(panel):
    """`events.data` fires twice per edit, so the handler has to be idempotent."""
    widget, viewer, bc = panel
    bc.pick()
    viewer.layers[BC_COORDINATES].data = np.array([[1.0, 2.0, 3.0]])
    once = widget._haemolynx_values()["inlet_node_coordinates"]

    bc.sync()
    bc.sync()

    assert widget._haemolynx_values()["inlet_node_coordinates"] == once


def test_editing_points_does_not_wipe_the_regions(panel):
    """The two layers share a sync; one must not clear the other's settings."""
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    bc.show()

    viewer.layers[BC_COORDINATES].data = np.array([[1.0, 2.0, 3.0]])

    assert widget._haemolynx_values()["outlet_node_volumes"] == [A_BOX]


def test_a_region_edited_in_the_viewer_reaches_the_setting(panel):
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    bc.show()

    layer = viewer.layers[regions_name("outlet")]
    corners, _ = rectangle_from_box([0.0, 10.0, 20.0], [100.0, 60.0, 90.0])
    layer.data = [corners]

    lo, hi = widget._haemolynx_values()["outlet_node_volumes"][0]
    assert lo[1:] == [10.0, 20.0] and hi[1:] == [60.0, 90.0]


def test_the_depth_slider_resizes_the_selected_region(panel):
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    bc.show()
    viewer.layers[regions_name("outlet")].selected_data = {0}

    bc.actions["outlet"].depth.value = 20.0

    lo, hi = widget._haemolynx_values()["outlet_node_volumes"][0]
    assert hi[0] - lo[0] == pytest.approx(20.0)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(50.0), "still centred where it was"


def test_clearing_a_roles_regions_leaves_the_other_roles_alone(panel):
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    rows_of(widget)["inlet_node_volumes"].value = [A_BOX]
    bc.show()

    bc.role.value = "outlet"
    bc.clear()

    assert widget._haemolynx_values()["outlet_node_volumes"] == []
    assert widget._haemolynx_values()["inlet_node_volumes"] == [A_BOX]


def test_assigning_a_selected_point_to_another_role_moves_it_between_settings(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[1.0, 2.0, 3.0]]
    bc.show()

    viewer.layers[BC_COORDINATES].selected_data = {0}
    bc.role.value = "venule_boundary"
    bc.assign()

    values = widget._haemolynx_values()
    assert values["inlet_node_coordinates"] == []
    assert values["venule_boundary_node_coordinates"] == [[1.0, 2.0, 3.0]]


# --- what cannot be done, said rather than silently failing ------------------


def test_snapping_before_a_run_says_why_rather_than_doing_nothing(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[1.0, 2.0, 3.0]]
    bc.show()

    bc.snap()

    report = widget._haemolynx_report()
    assert "run at least" in report
    assert "still correct" in report, "an unsnapped coordinate is not wrong"


def test_snapping_moves_a_coordinate_onto_a_terminal_node(panel):
    widget, viewer, bc = panel
    import networkx as nx

    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.array([50.0, 60.0, 70.0]))
    graph.add_edge(0, 1)
    bc.state.results = type("R", (), {"graph": graph})()

    rows_of(widget)["inlet_node_coordinates"].value = [[48.0, 61.0, 69.0]]
    bc.show()
    bc.snap()

    assert widget._haemolynx_values()["inlet_node_coordinates"] == [[50.0, 60.0, 70.0]]
    assert "um" in widget._haemolynx_report()


def test_picks_that_the_run_will_not_read_are_called_out(panel):
    """A role only reads its coordinates when its method says so."""
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"
    rows_of(widget)["inlet_node_coordinates"].value = [[1.0, 2.0, 3.0]]

    bc.show()

    assert "Not used" in widget._haemolynx_report()
    assert "edge_percent" in widget._haemolynx_report()


def test_a_config_that_cannot_be_read_is_reported_not_raised(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[1.0, 2.0]]

    bc.show()

    assert "could not be read" in widget._haemolynx_report()


# --- the panel is still the panel --------------------------------------------


def test_building_the_controls_writes_to_no_row(make_napari_viewer):
    """The controls must not quietly change the settings just by existing."""
    from haemolynx.pipeline import default_schema

    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)

    schema = default_schema()
    values = widget._haemolynx_values()
    for role_setting in ("inlet_node_coordinates", "outlet_node_volumes"):
        assert values[role_setting] == schema[role_setting].default


def test_the_controls_sit_on_the_boundaries_tab(panel):
    widget, viewer, bc = panel
    assert bc.widget is not None
    assert [str(choice) for choice in bc.role.choices][0] == "inlet"


# --- a region reads as the volume it is --------------------------------------


def test_a_region_is_drawn_as_a_translucent_solid_box(panel):
    """A rectangle on one slice says nothing about how deep the box goes, and
    a Shapes layer cannot draw a box in 3D -- so the box is a Surface."""
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]

    bc.show()

    regions = viewer.layers[regions_name("outlet")]
    assert list(regions.shape_type) == ["rectangle"], "only the editable handle"
    boxes = viewer.layers[boxes_name("outlet")]
    assert isinstance(boxes, napari.layers.Surface)
    vertices, faces = np.asarray(boxes.data[0]), np.asarray(boxes.data[1])
    assert vertices.shape == (8, 3) and faces.shape == (12, 3)
    assert vertices.min(axis=0) == pytest.approx(A_BOX[0])
    assert vertices.max(axis=0) == pytest.approx(A_BOX[1])
    assert boxes.opacity < 1.0, "translucent, so what is inside shows through"
    assert tuple(boxes.scale) == (1.0, 1.0, 1.0), "microns, like the settings"


def test_each_box_of_a_role_has_a_colour_of_its_own(panel):
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["outlet_node_volumes"].value = [
        A_BOX, [[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]], [[30.0, 0.0, 0.0], [50.0, 20.0, 20.0]],
    ]

    bc.show()

    boxes = viewer.layers[boxes_name("outlet")]
    colours = np.asarray(boxes.vertex_colors)
    per_box = [tuple(np.round(colours[8 * i], 4)) for i in range(3)]
    assert len(set(per_box)) == 3, "three boxes, three shades"
    for i in range(3):
        assert np.allclose(colours[8 * i: 8 * i + 8], colours[8 * i]), "one colour per box"


def test_the_boxes_survive_a_second_show(panel):
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    bc.show()

    rows_of(widget)["inlet_node_volumes"].value = [[[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]]
    bc.show()

    for role in ("inlet", "outlet"):
        assert list(viewer.layers[regions_name(role)].shape_type) == ["rectangle"]
        assert len(drawn_for(viewer, role)) == 8, "one box per role, in its own layer"


def test_reading_the_regions_back_counts_each_region_once(panel):
    """A sync that read the box back as well as the rectangle would multiply."""
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    bc.show()

    bc.sync()

    assert len(widget._haemolynx_values()["outlet_node_volumes"]) == 1


# --- the layers follow the form ----------------------------------------------


def test_editing_a_setting_redraws_the_layers(panel):
    """The form is one of the two authoritative directions, so it has to show."""
    widget, viewer, bc = panel
    bc.show()
    assert len(viewer.layers[BC_COORDINATES].data) == 0

    rows_of(widget)["inlet_node_coordinates"].value = [[10.0, 20.0, 30.0]]

    assert len(viewer.layers[BC_COORDINATES].data) == 1
    assert viewer.layers[BC_COORDINATES].data[0] == pytest.approx([10.0, 20.0, 30.0])


def test_changing_a_method_updates_what_the_panel_says_is_used(panel):
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    rows_of(widget)["outlet_node_selection_method"].value = "volume"
    bc.show()
    assert "Not used" not in widget._haemolynx_report()

    rows_of(widget)["outlet_node_selection_method"].value = "coordinates"

    assert "Not used" in widget._haemolynx_report()
    assert regions_name("outlet") in viewer.layers, "a configured region is never silently dropped"


def test_a_region_drawn_after_a_settings_edit_still_arrives(panel):
    """The redraw must leave the layer editable, not replace it with a picture."""
    widget, viewer, bc = panel
    bc.draw()                       # on the inlet page, the one open by default
    rows_of(widget)["inlet_node_volumes"].value = [A_BOX]

    regions = viewer.layers[regions_name("inlet")]
    assert regions.mode != "pan_zoom"


# --- only the settings the chosen method reads ------------------------------


def test_only_the_settings_the_chosen_method_reads_are_shown(panel):
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_selection_method"].value = "coordinates"
    rows_of(widget)["inlet_node_selection_method"].value = "volume"

    visible = bc.state.visible
    assert "outlet_node_coordinates" in visible
    assert "inlet_node_volumes" in visible
    assert "outlet_node_volumes" not in visible
    assert "inlet_node_coordinates" not in visible
    # `.visible` on a row of a tab that is not on screen reads False whatever
    # was set, so what the panel hid is recorded rather than queried.
    assert "outlet_node_volumes" in bc.state.hidden
    assert "outlet_node_coordinates" not in bc.state.hidden


def test_a_method_row_is_never_hidden(panel):
    """It is the row that decides which of the others you need."""
    widget, viewer, bc = panel
    # `all_degree_1` takes every terminal, so it configures nothing at all --
    # the case where hiding everything would hide the way back.
    rows_of(widget)["outlet_node_selection_method"].value = "all_degree_1"

    assert "outlet_node_selection_method" not in bc.state.hidden
    assert "outlet_node_coordinates" in bc.state.hidden


def test_editing_one_role_does_not_wipe_another(panel):
    """Applying a layer fires its own data event.

    Unguarded, that event reads the half-written layer straight back into the
    settings, and the role whose features had not been written yet loses its
    boxes -- silently, since the row it emptied is the one you were not
    looking at.
    """
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    bc.show()

    rows_of(widget)["inlet_node_volumes"].value = [[[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]]

    values = widget._haemolynx_values()
    assert len(values["outlet_node_volumes"]) == 1
    assert len(values["inlet_node_volumes"]) == 1
    assert len(viewer.layers[regions_name("outlet")].data) == 1
    assert len(viewer.layers[regions_name("inlet")].data) == 1
    assert len(drawn_for(viewer, "outlet")) == len(drawn_for(viewer, "inlet")) == 8


def test_a_setting_sits_below_the_method_that_asks_for_it(panel):
    widget, viewer, bc = panel
    names = list(rows_of(widget))
    order = bc.row_order([n for n in names if n.startswith(("inlet_", "outlet_"))])

    assert order.index("outlet_node_selection_method") < order.index("outlet_node_coordinates")
    assert order.index("inlet_node_volumes") < order.index("outlet_node_selection_method")


def test_the_ordering_keeps_every_row_it_was_given(panel):
    """A row this module does not know about must not disappear off the tab."""
    widget, viewer, bc = panel
    names = list(rows_of(widget))

    assert sorted(bc.row_order(names)) == sorted(names)


# --- one sub-tab per role ----------------------------------------------------


def test_there_is_one_sub_tab_per_role(panel):
    widget, viewer, bc = panel
    tabs = bc.state.tabs

    assert tabs.count() == len(ROLES)
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Inlet", "Outlet", "Arteriole", "Venule",
        "Large vessel inlet", "Large vessel outlet",
    ]


def _role_tab_enabled(bc):
    tabs = bc.state.tabs
    return {ROLES[i]: tabs.isTabEnabled(i) for i in range(tabs.count())}


def test_large_auto_greys_inlet_and_outlet_role_tabs(panel):
    """Large-vessel automated assignment disables Inlet/Outlet; keeps A/V."""
    widget, viewer, bc = panel
    rows = rows_of(widget)
    tabs = bc.state.tabs

    rows["automated_vessel_assignment"].value = True
    rows["use_large_vessel_masks"].value = True
    rows["use_small_vessel_masks_for_boundary_assignment"].value = False

    enabled = _role_tab_enabled(bc)
    assert enabled["inlet"] is False
    assert enabled["outlet"] is False
    assert enabled["arteriole_boundary"] is True
    assert enabled["venule_boundary"] is True
    # Grey out, do not hide: tab bar entries remain present.
    assert tabs.isTabVisible(0) is True
    assert tabs.isTabVisible(1) is True
    assert tabs.count() == len(ROLES)


def test_small_auto_greys_arteriole_and_venule_role_tabs(panel):
    widget, viewer, bc = panel
    rows = rows_of(widget)
    tabs = bc.state.tabs

    rows["automated_vessel_assignment"].value = True
    rows["use_small_vessel_masks_for_boundary_assignment"].value = True

    enabled = _role_tab_enabled(bc)
    assert enabled["arteriole_boundary"] is False
    assert enabled["venule_boundary"] is False
    # Automated (large) also greys inlet/outlet when the root toggle is on.
    assert enabled["inlet"] is False
    assert enabled["outlet"] is False
    assert tabs.isTabVisible(2) is True
    assert tabs.isTabVisible(3) is True
    assert tabs.count() == len(ROLES)


def test_both_autos_off_enables_every_role_tab(panel):
    """large_vessel_inlet/outlet stay disabled here -- they gate on
    assign_large_vessel_branch_orders, not either of these two flags; see
    test_large_vessel_network_mode_toggles_the_new_role_tabs below."""
    widget, viewer, bc = panel
    rows = rows_of(widget)

    rows["automated_vessel_assignment"].value = False
    rows["use_small_vessel_masks_for_boundary_assignment"].value = False

    enabled = _role_tab_enabled(bc)
    assert enabled == {
        "inlet": True,
        "outlet": True,
        "arteriole_boundary": True,
        "venule_boundary": True,
        "large_vessel_inlet": False,
        "large_vessel_outlet": False,
    }


def test_large_vessel_network_mode_toggles_the_new_role_tabs(panel):
    """Inverted polarity: greyed while the feature is off, enabled once on --
    the opposite of the automated-assignment gates above."""
    widget, viewer, bc = panel
    rows = rows_of(widget)

    enabled = _role_tab_enabled(bc)
    assert enabled["large_vessel_inlet"] is False
    assert enabled["large_vessel_outlet"] is False

    rows["assign_large_vessel_branch_orders"].value = True

    enabled = _role_tab_enabled(bc)
    assert enabled["large_vessel_inlet"] is True
    assert enabled["large_vessel_outlet"] is True


def test_choosing_a_sub_tab_chooses_the_role(panel):
    """The tab is the role: two ways to say it could disagree."""
    widget, viewer, bc = panel

    bc.state.tabs.setCurrentIndex(1)

    assert str(bc.role.value) == "outlet"


def test_choosing_the_role_moves_the_sub_tab(panel):
    widget, viewer, bc = panel

    bc.role.value = "venule_boundary"

    assert bc.state.tabs.currentIndex() == 3


def test_a_role_only_shows_its_own_settings(panel):
    """The complaint this answers: four near-identical methods on one page."""
    widget, viewer, bc = panel
    inlet_page = bc.state.tabs.widget(0)
    natives = set(inlet_page.findChildren(type(rows_of(widget)[
        "inlet_node_selection_method"].native)))

    assert rows_of(widget)["inlet_node_selection_method"].native in natives
    assert rows_of(widget)["outlet_node_selection_method"].native not in natives


def test_a_picked_point_takes_the_role_of_the_open_sub_tab(panel):
    widget, viewer, bc = panel
    bc.show()
    bc.state.tabs.setCurrentIndex(1)

    points = viewer.layers[BC_COORDINATES]
    points.add([[7.0, 8.0, 9.0]])

    assert widget._haemolynx_values()["outlet_node_coordinates"] == [[7.0, 8.0, 9.0]]
    assert widget._haemolynx_values()["inlet_node_coordinates"] == []


def page_of(bc, widget_row):
    """Which role's page a row is sitting on, if any."""
    for role, holder in bc.holders.items():
        if widget_row in list(holder):
            return role
    return None


def test_a_shared_row_sits_under_the_role_that_is_reading_it(panel):
    """There is one axis row and Qt gives it one parent, so it cannot be on
    all four pages -- it goes to the page being looked at instead."""
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"

    assert page_of(bc, rows_of(widget)["boundary_axis"]) == "inlet"
    assert page_of(bc, rows_of(widget)["boundary_first_percent"]) == "inlet"


def test_a_shared_row_follows_the_open_sub_tab(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"
    rows_of(widget)["outlet_node_selection_method"].value = "edge_percent"

    bc.state.tabs.setCurrentIndex(1)

    assert page_of(bc, rows_of(widget)["boundary_axis"]) == "outlet"


def test_a_role_reads_the_percentage_for_its_own_end(panel):
    """A run computes both ends every time and a role takes one of them, so
    showing an inlet the outlet percentage invites setting a dead number."""
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"

    assert page_of(bc, rows_of(widget)["boundary_first_percent"]) == "inlet"
    assert page_of(bc, rows_of(widget)["boundary_last_percent"]) is None

    bc.state.tabs.setCurrentIndex(1)
    rows_of(widget)["outlet_node_selection_method"].value = "edge_percent"

    assert page_of(bc, rows_of(widget)["boundary_last_percent"]) == "outlet"
    assert page_of(bc, rows_of(widget)["boundary_first_percent"]) is None


def test_a_shared_row_leaves_when_the_method_stops_reading_it(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"

    rows_of(widget)["inlet_node_selection_method"].value = "coordinates"

    assert page_of(bc, rows_of(widget)["boundary_axis"]) is None


def test_a_shared_row_is_never_on_two_pages(panel):
    """The whole reason it moves rather than being copied."""
    widget, viewer, bc = panel
    for name in ROLES:
        rows_of(widget)[method_setting(name)].value = "edge_percent"

    for name in ("boundary_axis", "boundary_first_percent", "boundary_last_percent"):
        homes = [role for role in ROLES
                 if rows_of(widget)[name] in list(bc.holders[role])]
        assert len(homes) <= 1


# --- the depth slider on a real, anisotropic stack ---------------------------


def test_showing_survives_a_stack_whose_depth_the_slider_rounds(panel):
    """`FloatSlider` rounds the maximum it is given and raises on a value past
    it, so defaulting the depth to the stack's exact span blew up on any stack
    whose depth was not a round number -- which is any anisotropic one."""
    widget, viewer, bc = panel

    bc.show()

    span = float(viewer.layers["stack"].extent.world[1][0]
                 - viewer.layers["stack"].extent.world[0][0])
    assert bc.depth_slider().value == pytest.approx(span, abs=1e-3)
    assert bc.depth_slider().value <= bc.depth_slider().max


# --- coordinates that do not lie on the image --------------------------------


def test_a_coordinate_off_the_image_is_reported_with_the_images_size(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[5000.0, 5.0, 5.0]]

    bc.show()

    report = widget._haemolynx_report()
    assert "Outside the image" in report
    assert "microns, not voxel indices" in report


def test_a_coordinate_on_the_image_is_not_reported(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[10.0, 20.0, 30.0]]

    bc.show()

    assert "Outside the image" not in widget._haemolynx_report()


# --- a role's controls live on that role's page ------------------------------


CONTROLS = ("pick", "draw", "depth", "move", "assign", "clear")


@pytest.mark.parametrize("control", CONTROLS)
def test_each_role_has_its_own_copy_of_a_control(panel, control):
    """A control that acts on "the chosen role" from another role's page is a
    control that can be pressed by mistake; one per page cannot be."""
    widget, viewer, bc = panel
    natives = {getattr(bc.actions[role], control).native for role in ROLES}

    assert len(natives) == len(ROLES)
    page = bc.state.tabs.widget(1)
    own = getattr(bc.actions["outlet"], control).native
    assert own in page.findChildren(type(own))
    assert getattr(bc.actions["inlet"], control).native not in page.findChildren(type(own))


def test_the_tab_level_controls_are_the_ones_that_are_not_per_role(panel):
    widget, viewer, bc = panel
    labels = [getattr(w, "text", "") for w in bc.widget]

    assert labels == ["Show these boundary conditions",
                      "Snap selected to nearest terminal"]


def test_pressing_a_control_acts_on_the_page_it_sits_on(panel):
    """Even from another tab: the page it is on is the answer to "which role"."""
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_selection_method"].value = "volume"
    bc.state.tabs.setCurrentIndex(0)

    bc.actions["outlet"].draw.changed()

    assert str(bc.role.value) == "outlet"
    viewer.layers[regions_name("outlet")].add_rectangles(
        np.array([[10.0, 5.0, 5.0], [10.0, 5.0, 25.0],
                  [10.0, 25.0, 25.0], [10.0, 25.0, 5.0]])
    )
    assert len(widget._haemolynx_values()["outlet_node_volumes"]) == 1
    assert widget._haemolynx_values()["inlet_node_volumes"] == []


def test_a_method_only_shows_the_controls_it_can_use(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "coordinates"
    rows_of(widget)["outlet_node_selection_method"].value = "volume"
    rows_of(widget)["venule_boundary_selection_method"].value = "edge_percent"

    assert bc.state.actions["inlet"] == {"pick", "move", "assign"}
    # A volume role inserts a box rather than drawing one.
    assert bc.state.actions["outlet"] == {"insert_box", "box_choice", "box_size", "box_step", "box_move", "box_nodes", "use_node", "remove_box"} | {"clear"}
    assert bc.state.actions["venule_boundary"] == set(), "nothing to point at"


def test_each_role_keeps_its_own_region_depth(panel):
    widget, viewer, bc = panel
    bc.show()

    bc.actions["outlet"].depth.value = 12.0

    assert bc.actions["inlet"].depth.value != pytest.approx(12.0)
    bc.role.value = "outlet"
    assert bc.depth_slider().value == pytest.approx(12.0)


def test_defaulting_the_depths_does_not_move_the_role(panel):
    """Every slider is written when the range is set, and each is wired to its
    own role -- so an unguarded write walked the role to the last one."""
    widget, viewer, bc = panel

    bc.show()

    assert str(bc.role.value) == "inlet"
    assert bc.state.tabs.currentIndex() == 0


# --- drawing into a layer that already holds a region ------------------------


def test_a_region_drawn_into_an_empty_layer_reaches_its_role(panel):
    """The first thing anyone does, and it had two faults at once: an empty
    feature column is float64, so the role default came back NaN and made a
    NaN box that no settings row could ever parse again."""
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_selection_method"].value = "volume"

    bc.actions["outlet"].draw.changed()
    viewer.layers[regions_name("outlet")].add_rectangles(
        np.array([[10.0, 5.0, 5.0], [10.0, 5.0, 25.0],
                  [10.0, 25.0, 25.0], [10.0, 25.0, 5.0]])
    )

    boxes = widget._haemolynx_values()["outlet_node_volumes"]
    assert len(boxes) == 1
    assert np.isfinite(np.asarray(boxes, dtype=float)).all(), "a NaN corner is a lost box"
    assert ast.literal_eval(str(boxes)) == boxes, "the row must survive being read back"


def test_redrawing_over_a_hand_drawn_region_keeps_it(panel):
    """Showing again must neither lose a hand-drawn region nor add to it."""
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["outlet_node_selection_method"].value = "volume"
    bc.actions["outlet"].draw.changed()
    viewer.layers[regions_name("outlet")].add_rectangles(
        np.array([[10.0, 5.0, 5.0], [10.0, 5.0, 25.0],
                  [10.0, 25.0, 25.0], [10.0, 25.0, 5.0]])
    )
    before = widget._haemolynx_values()["outlet_node_volumes"]

    bc.show()

    assert len(viewer.layers[regions_name("outlet")].data) == 1
    assert len(drawn_for(viewer, "outlet")) == 8
    assert widget._haemolynx_values()["outlet_node_volumes"] == before


# --- the depth slider is a resize, not just a default ------------------------


def draw_a_region(widget, viewer, bc, *, role="outlet", z=10.0):
    no_bands(widget)
    rows_of(widget)[method_setting(role)].value = "volume"
    bc.actions[role].draw.changed()
    viewer.layers[regions_name(role)].add_rectangles(
        np.array([[z, 5.0, 5.0], [z, 5.0, 25.0], [z, 25.0, 25.0], [z, 25.0, 5.0]])
    )


def test_the_depth_slider_resizes_this_roles_regions(panel):
    """Nothing selected used to mean "the size for the next one", which reads
    as the slider doing nothing at all."""
    widget, viewer, bc = panel
    draw_a_region(widget, viewer, bc)

    bc.actions["outlet"].depth.value = 6.0

    box = widget._haemolynx_values()["outlet_node_volumes"][0]
    assert box[0][0] == pytest.approx(7.0)
    assert box[1][0] == pytest.approx(13.0)


def test_the_drawn_box_follows_the_slider(panel):
    """The box is drawn from the settings, so it only moves once they do."""
    widget, viewer, bc = panel
    draw_a_region(widget, viewer, bc)

    bc.actions["outlet"].depth.value = 6.0

    drawn = drawn_for(viewer, "outlet")
    assert drawn[:, 0].min() == pytest.approx(7.0)
    assert drawn[:, 0].max() == pytest.approx(13.0)


def test_the_slider_leaves_another_roles_regions_alone(panel):
    widget, viewer, bc = panel
    draw_a_region(widget, viewer, bc, role="outlet", z=10.0)
    draw_a_region(widget, viewer, bc, role="inlet", z=20.0)
    before = widget._haemolynx_values()["outlet_node_volumes"]

    bc.actions["inlet"].depth.value = 4.0

    assert widget._haemolynx_values()["outlet_node_volumes"] == before
    starting = widget._haemolynx_values()["inlet_node_volumes"][0]
    assert starting[1][0] - starting[0][0] == pytest.approx(4.0)


def test_a_selected_region_is_the_one_that_resizes(panel):
    """So two regions of one role can differ."""
    widget, viewer, bc = panel
    draw_a_region(widget, viewer, bc, z=10.0)
    draw_a_region(widget, viewer, bc, z=30.0)
    viewer.layers[regions_name("outlet")].selected_data = {0}

    bc.actions["outlet"].depth.value = 8.0

    depths = [box[1][0] - box[0][0]
              for box in widget._haemolynx_values()["outlet_node_volumes"]]
    assert sorted(round(d, 3) for d in depths)[0] == pytest.approx(8.0)
    assert len({round(d, 3) for d in depths}) == 2


# --- moving what you picked --------------------------------------------------


def test_move_puts_the_layer_into_naparis_select_tool(panel):
    """Placing and moving are two modes of one layer, and clicking in `add`
    mode makes another point rather than picking up the one under it."""
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "coordinates"
    bc.actions["inlet"].pick.changed()
    assert viewer.layers[BC_COORDINATES].mode == "add"

    bc.actions["inlet"].move.changed()

    assert viewer.layers[BC_COORDINATES].mode == "select"
    assert "drag to move it" in widget._haemolynx_report()


def test_dragging_a_coordinate_writes_where_it_landed(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "coordinates"
    rows_of(widget)["inlet_node_coordinates"].value = [[10.0, 8.0, 9.0]]
    bc.show()
    points = viewer.layers[BC_COORDINATES]

    points.selected_data = {0}
    points._move({0}, [10.0, 8.0, 9.0])      # press: records the drag origin
    points._move({0}, [10.0, 18.0, 19.0])    # and drag

    assert widget._haemolynx_values()["inlet_node_coordinates"] == [
        [10.0, 18.0, 19.0]
    ]


def test_move_on_a_region_role_reaches_for_the_regions(panel):
    widget, viewer, bc = panel
    draw_a_region(widget, viewer, bc)

    bc.actions["outlet"].move.changed()

    assert viewer.layers[regions_name("outlet")].mode == "select"


def test_regions_cannot_be_moved_in_the_3d_view(panel):
    widget, viewer, bc = panel
    draw_a_region(widget, viewer, bc)
    viewer.dims.ndisplay = 3

    bc.actions["outlet"].move.changed()

    assert "2D" in widget._haemolynx_report()


def test_move_before_anything_is_drawn_says_so(panel):
    widget, viewer, bc = panel

    bc.actions["inlet"].move.changed()

    assert "Nothing to move yet" in widget._haemolynx_report()


# --- showing what edge_percent selects from ----------------------------------


def a_graph_spanning(low, high, axis=1):
    import networkx as nx

    graph = nx.MultiGraph()
    first, second = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    first[axis], second[axis] = low, high
    graph.add_node(0, pos=np.array(first))
    graph.add_node(1, pos=np.array(second))
    graph.add_edge(0, 1)
    return graph


def test_edge_percent_draws_the_band_it_will_select_from(panel):
    """The one method whose region is implied rather than written down: before
    this there was nothing on screen until a run had already used it."""
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"
    rows_of(widget)["boundary_first_percent"].value = 10.0
    rows_of(widget)["boundary_axis"].value = 1

    bc.show()

    assert boxes_name("inlet") in viewer.layers
    drawn = drawn_for(viewer, "inlet")
    image_y = float(viewer.layers["stack"].extent.world[1][1])
    assert drawn[:, 1].min() == pytest.approx(0.0)
    assert drawn[:, 1].max() == pytest.approx(image_y * 0.1, rel=1e-3)


def test_the_outlet_band_sits_at_the_far_end(panel):
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_selection_method"].value = "edge_percent"
    rows_of(widget)["boundary_last_percent"].value = 20.0

    bc.show()

    drawn = drawn_for(viewer, "outlet")
    image_y = float(viewer.layers["stack"].extent.world[1][1])
    assert drawn[:, 1].max() == pytest.approx(image_y)
    assert drawn[:, 1].min() == pytest.approx(image_y * 0.8, rel=1e-3)


def test_the_band_says_which_span_it_was_drawn_across(panel):
    """A run measures across the terminals, so the pre-run band is a guess."""
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"

    bc.show()
    assert "across the image" in widget._haemolynx_report()
    assert "3. Graph" in widget._haemolynx_report()

    bc.state.results = type("R", (), {"graph": a_graph_spanning(40.0, 160.0)})()
    bc.show()
    assert "across the terminals" in widget._haemolynx_report()


def test_the_band_follows_the_terminals_once_there_are_some(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"
    rows_of(widget)["boundary_first_percent"].value = 50.0
    bc.state.results = type("R", (), {"graph": a_graph_spanning(40.0, 160.0)})()

    bc.show()

    drawn = drawn_for(viewer, "inlet")
    assert drawn[:, 1].min() == pytest.approx(40.0)
    assert drawn[:, 1].max() == pytest.approx(100.0)


def test_the_band_follows_the_percentage_as_it_is_typed(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"
    bc.show()

    rows_of(widget)["boundary_first_percent"].value = 40.0

    drawn = drawn_for(viewer, "inlet")
    image_y = float(viewer.layers["stack"].extent.world[1][1])
    assert drawn[:, 1].max() == pytest.approx(image_y * 0.4, rel=1e-3)


def test_a_band_is_never_read_back_as_a_configured_region(panel):
    """It is what a percentage works out to, not something anyone typed."""
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"
    bc.show()

    bc.sync()

    assert widget._haemolynx_values()["inlet_node_volumes"] == []
    assert widget._haemolynx_values()["outlet_node_volumes"] == []


def test_a_band_goes_away_with_the_method_that_made_it(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_selection_method"].value = "edge_percent"
    bc.show()
    assert len(drawn_for(viewer, "inlet")) == 8, "one box"
    assert regions_name("inlet") not in viewer.layers, "and nothing to edit"

    rows_of(widget)["inlet_node_selection_method"].value = "all_degree_1"

    assert not len(drawn_for(viewer, "inlet"))
    assert boxes_name("inlet") not in viewer.layers


def test_a_shared_row_with_no_page_is_not_a_window(panel):
    """A magicgui row removed from its container has no Qt parent, and a
    visible widget with no parent is a top-level window: "Boundary last
    percent (percent)", floating on its own next to napari."""
    from qtpy.QtWidgets import QApplication

    widget, viewer, bc = panel
    widget.show()
    before = {id(w) for w in QApplication.topLevelWidgets() if w.isVisible()}

    for role in ROLES:
        rows_of(widget)[method_setting(role)].value = "edge_percent"
    for role in ROLES:
        rows_of(widget)[method_setting(role)].value = "coordinates"

    appeared = [w for w in QApplication.topLevelWidgets()
                if w.isVisible() and id(w) not in before]
    assert appeared == []
    assert rows_of(widget)["boundary_last_percent"].visible is False


def test_an_inlet_ring_is_green_and_an_outlet_ring_is_red(panel):
    widget, viewer, bc = panel
    rows_of(widget)["inlet_node_coordinates"].value = [[10.0, 20.0, 30.0]]
    rows_of(widget)["outlet_node_coordinates"].value = [[40.0, 50.0, 60.0]]

    bc.show()

    colours = viewer.layers[BC_COORDINATES].face_color
    expected = dict(role_colours())
    assert tuple(colours[0]) == pytest.approx(expected["inlet"])
    assert tuple(colours[1]) == pytest.approx(expected["outlet"])


# --- a layer per role, in that role's colour ---------------------------------


def test_an_inlet_region_is_green_and_an_outlet_region_is_red(panel):
    """Both attributes, not just the face: a line has no face, so an outline
    drawn as twelve line shapes kept napari's default white however the faces
    were coloured."""
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["inlet_node_volumes"].value = [A_BOX]
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]

    bc.show()

    expected = dict(role_colours())
    for role in ("inlet", "outlet"):
        layer = viewer.layers[regions_name(role)]
        for attribute in ("edge_color", "face_color"):
            colours = np.unique(getattr(layer, attribute), axis=0)
            assert len(colours) == 1, f"{role} {attribute} is not one colour"
            assert tuple(colours[0]) == pytest.approx(expected[role])


def test_a_roles_regions_can_be_hidden_on_their_own(panel):
    """The reason for splitting them: one visibility per layer."""
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["inlet_node_volumes"].value = [A_BOX]
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    bc.show()

    viewer.layers[regions_name("inlet")].visible = False

    assert viewer.layers[regions_name("outlet")].visible is True


def test_a_hidden_role_stays_hidden_across_a_redraw(panel):
    """`_add_or_update` keeps what the user set; a fresh layer would not."""
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["inlet_node_volumes"].value = [A_BOX]
    bc.show()
    viewer.layers[regions_name("inlet")].visible = False

    bc.show()

    assert viewer.layers[regions_name("inlet")].visible is False


def test_a_role_with_nothing_left_loses_its_layer(panel):
    """`_apply_layers` only adds and updates, so the layer would linger."""
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["inlet_node_volumes"].value = [A_BOX]
    bc.show()
    assert regions_name("inlet") in viewer.layers

    rows_of(widget)["inlet_node_volumes"].value = []

    assert regions_name("inlet") not in viewer.layers


def test_the_layer_being_drawn_into_is_never_taken_away(panel):
    """Draw makes an empty layer to draw into, and the next redraw would find
    a role with nothing to show and remove it from under the tool."""
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["inlet_node_selection_method"].value = "volume"
    bc.actions["inlet"].draw.changed()

    rows_of(widget)["outlet_node_coordinates"].value = [[1.0, 2.0, 3.0]]

    assert regions_name("inlet") in viewer.layers
    assert viewer.layers[regions_name("inlet")].mode == "add_rectangle"


def test_reassigning_a_region_moves_it_to_the_other_roles_layer(panel):
    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["inlet_node_volumes"].value = [A_BOX]
    bc.show()
    inlet = viewer.layers[regions_name("inlet")]
    inlet.selected_data = {0}

    bc.role.value = "outlet"
    bc.assign()

    assert widget._haemolynx_values()["inlet_node_volumes"] == []
    assert len(widget._haemolynx_values()["outlet_node_volumes"]) == 1
    assert regions_name("outlet") in viewer.layers


# --- several regions, drawn with napari's own tool ---------------------------


def start_drawing(widget, viewer, bc, role="outlet"):
    no_bands(widget)
    rows_of(widget)[method_setting(role)].value = "volume"
    bc.actions[role].draw.changed()
    return viewer.layers[regions_name(role)], float(viewer.dims.point[0])


def test_drawing_a_second_region_keeps_the_first_and_adds_the_second(panel):
    """The reported failure. A drag callback redrew the layer from the settings
    when the mouse came up -- before napari's rectangle tool had finished, so
    the rectangle just drawn was wiped, and the tool was left holding a
    selection whose box no longer existed (`_selected_box` None, the TypeError
    on the next drag)."""
    widget, viewer, bc = panel
    regions, z = start_drawing(widget, viewer, bc)

    drag(regions, (z, 10.0, 10.0), (z, 50.0, 60.0))
    drag(regions, (z, 100.0, 100.0), (z, 150.0, 160.0))
    drag(regions, (z, 60.0, 170.0), (z, 90.0, 200.0))

    boxes = widget._haemolynx_values()["outlet_node_volumes"]
    assert len(boxes) == 3
    assert [box[0][1:] for box in boxes] == [[10.0, 10.0], [100.0, 100.0], [60.0, 170.0]]
    assert len(regions.data) == 3
    assert regions.mode == "add_rectangle", "still drawing: a fourth is one drag away"


def test_the_rectangle_just_drawn_is_left_as_napari_left_it(panel):
    """Rewriting the layer the tool works on resets its selection underneath
    it; a layer that already says what the settings say is not rewritten."""
    widget, viewer, bc = panel
    regions, z = start_drawing(widget, viewer, bc)
    drag(regions, (z, 10.0, 10.0), (z, 50.0, 60.0))
    drag(regions, (z, 100.0, 100.0), (z, 150.0, 160.0))

    assert regions.selected_data == {1}, "napari selects what it has just drawn"
    assert regions._selected_box is not None


def test_every_region_drawn_shows_as_its_own_box(panel):
    widget, viewer, bc = panel
    regions, z = start_drawing(widget, viewer, bc)

    drag(regions, (z, 10.0, 10.0), (z, 50.0, 60.0))
    drag(regions, (z, 100.0, 100.0), (z, 150.0, 160.0))

    vertices = drawn_for(viewer, "outlet")
    assert len(vertices) == 16, "two boxes, eight corners each"
    colours = np.asarray(viewer.layers[boxes_name("outlet")].vertex_colors)
    assert not np.allclose(colours[0], colours[8]), "and each in its own shade"


def test_the_boxes_appearing_do_not_take_the_tool_away(panel):
    """napari makes a newly added layer the active one, and mouse events go to
    the active layer: the first box appearing would have ended the drawing."""
    widget, viewer, bc = panel
    regions, z = start_drawing(widget, viewer, bc)

    drag(regions, (z, 10.0, 10.0), (z, 50.0, 60.0))

    assert viewer.layers.selection.active is regions


def test_a_boundary_edit_does_not_rebuild_the_rest_of_the_view(panel, monkeypatch):
    """What made picking lag: every edit went through `_apply_layers`, which
    also rebuilds the vessel tube mesh and re-runs the Z-depth filter."""
    import haemolynx.gui._widget as panel_module

    calls = []
    monkeypatch.setattr(panel_module, "_sync_vessel_tubes",
                        lambda *a, **k: calls.append("tubes"))
    widget, viewer, bc = panel
    regions, z = start_drawing(widget, viewer, bc)

    drag(regions, (z, 10.0, 10.0), (z, 50.0, 60.0))
    rows_of(widget)["outlet_node_volumes"].value = [A_BOX]
    bc.show()

    assert calls == []


def test_clearing_while_drawing_empties_the_layer_rather_than_keeping_it(panel):
    """The layer a tool is on is kept -- but emptied, or the next edit would
    read its rectangles straight back into the setting that was just cleared."""
    widget, viewer, bc = panel
    regions, z = start_drawing(widget, viewer, bc)
    drag(regions, (z, 10.0, 10.0), (z, 50.0, 60.0))

    bc.actions["outlet"].clear.changed()
    drag(regions, (z, 100.0, 100.0), (z, 150.0, 160.0))

    boxes = widget._haemolynx_values()["outlet_node_volumes"]
    assert len(boxes) == 1 and boxes[0][0][1:] == [100.0, 100.0]


# --- drawing a region in the 3D view -----------------------------------------

#: napari's default 3D camera looks down z with y up the screen.
VIEW_DOWN_Z = {"view": (1.0, 0.0, 0.0), "up": (0.0, -1.0, 0.0), "displayed": (0, 1, 2)}


def start_drawing_in_3d(widget, viewer, bc, role="outlet"):
    no_bands(widget)
    rows_of(widget)[method_setting(role)].value = "volume"
    viewer.dims.ndisplay = 3
    bc.actions[role].draw.changed()
    return viewer.layers[regions_name(role)]


def test_draw_is_offered_in_the_3d_view(panel):
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_selection_method"].value = "volume"

    viewer.dims.ndisplay = 3

    assert bc.actions["outlet"].draw.enabled is True


def test_a_drag_in_3d_adds_a_box_through_the_whole_stack(panel):
    widget, viewer, bc = panel
    regions = start_drawing_in_3d(widget, viewer, bc)

    drag(regions, (59.0, 20.0, 30.0), (59.0, 80.0, 120.0), **VIEW_DOWN_Z)

    (lo, hi), = widget._haemolynx_values()["outlet_node_volumes"]
    extent = viewer.layers["stack"].extent.world
    assert lo[1:] == pytest.approx([20.0, 30.0]) and hi[1:] == pytest.approx([80.0, 120.0])
    assert lo[0] == pytest.approx(float(extent[0][0]), abs=1e-3)
    assert hi[0] == pytest.approx(float(extent[1][0]), abs=1e-3)
    assert len(drawn_for(viewer, "outlet")) == 8, "and shows as a box"


def test_several_boxes_can_be_drawn_in_3d(panel):
    widget, viewer, bc = panel
    regions = start_drawing_in_3d(widget, viewer, bc)
    drag(regions, (59.0, 20.0, 30.0), (59.0, 80.0, 120.0), **VIEW_DOWN_Z)

    bc.actions["outlet"].draw.changed()
    drag(regions, (59.0, 120.0, 130.0), (59.0, 180.0, 200.0), **VIEW_DOWN_Z)

    assert len(widget._haemolynx_values()["outlet_node_volumes"]) == 2
    assert len(drawn_for(viewer, "outlet")) == 16


def test_the_view_holds_still_while_a_box_is_dragged_out(panel):
    """The camera follows the active layer's `mouse_pan`; a drag that also
    turned the view would draw a box nobody aimed."""
    widget, viewer, bc = panel
    regions = start_drawing_in_3d(widget, viewer, bc)

    assert viewer.layers.selection.active is regions
    assert regions.mouse_pan is False

    drag(regions, (59.0, 20.0, 30.0), (59.0, 80.0, 120.0), **VIEW_DOWN_Z)

    assert regions.mouse_pan is True, "the mouse goes back to the camera"
    assert bc.draw_in_3d not in regions.mouse_drag_callbacks


def test_a_click_in_3d_without_a_drag_adds_nothing_and_says_so(panel):
    widget, viewer, bc = panel
    regions = start_drawing_in_3d(widget, viewer, bc)

    drag(regions, (59.0, 20.0, 30.0), (59.0, 20.0, 30.0), **VIEW_DOWN_Z)

    assert widget._haemolynx_values()["outlet_node_volumes"] == []
    assert "no box" in widget._haemolynx_report()
    assert regions.mouse_pan is True


def test_going_back_to_2d_gives_the_mouse_back(panel):
    widget, viewer, bc = panel
    regions = start_drawing_in_3d(widget, viewer, bc)

    viewer.dims.ndisplay = 2

    assert bc.state.draw3d is None
    assert bc.draw_in_3d not in regions.mouse_drag_callbacks


def test_a_box_drawn_in_3d_can_be_trimmed_with_the_depth_slider(panel):
    widget, viewer, bc = panel
    regions = start_drawing_in_3d(widget, viewer, bc)
    drag(regions, (59.0, 20.0, 30.0), (59.0, 80.0, 120.0), **VIEW_DOWN_Z)

    bc.actions["outlet"].depth.value = 20.0

    (lo, hi), = widget._haemolynx_values()["outlet_node_volumes"]
    assert hi[0] - lo[0] == pytest.approx(20.0)
    assert lo[1:] == pytest.approx([20.0, 30.0]), "across the screen it stays put"


# --- node IDs, picked by clicking the graph -----------------------------------


#: A Y on the z = 20 um plane: terminal 0, junction 1, terminals 2 and 3.
Y_NODES = {0: (20.0, 40.0, 100.0), 1: (20.0, 80.0, 100.0),
           2: (20.0, 120.0, 60.0), 3: (20.0, 120.0, 140.0)}


def with_graph(viewer, bc):
    """The nodes and vessels layers a '3. Graph' run leaves, from the real converter."""
    import networkx as nx
    from types import SimpleNamespace

    from haemolynx.gui._widget import _apply_layers
    from haemolynx.gui.results import ResultLayers

    graph = nx.MultiGraph()
    for node, pos in Y_NODES.items():
        graph.add_node(node, pos=np.asarray(pos))
    for u, v in ((0, 1), (1, 2), (1, 3)):
        graph.add_edge(u, v, key=0, length=1.0, voxels=[list(Y_NODES[u]), list(Y_NODES[v])])
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished("build_network", SimpleNamespace(
        graph=graph,
        volume=SimpleNamespace(image=None, skeleton=None, voxel_size_xyz=(1.0, 1.0, 1.0),
                               voxel_size_zyx=(1.0, 1.0, 1.0)),
        large_arteriole_mask=None, large_venule_mask=None,
        small_arteriole_mask=None, small_venule_mask=None,
    )))
    # The real converter, not a stand-in: what the tab reads off it is `graph`.
    bc.state.results = results
    viewer.dims.ndisplay = 2
    viewer.dims.set_point(0, 20.0)
    return graph


def click(viewer, position, *, drag_to_pixel=None):
    """A left press and release through napari's own viewer dispatch."""
    from napari.utils._test_utils import read_only_mouse_event
    from napari.utils.interactions import (
        mouse_move_callbacks, mouse_press_callbacks, mouse_release_callbacks,
    )

    common = {"button": 1, "dims_displayed": [1, 2]}
    mouse_press_callbacks(viewer, read_only_mouse_event(
        type="mouse_press", position=tuple(position), pos=np.array([100.0, 100.0]),
        **common))
    if drag_to_pixel is not None:
        mouse_move_callbacks(viewer, read_only_mouse_event(
            type="mouse_move", is_dragging=True, position=tuple(position),
            pos=np.asarray(drag_to_pixel, dtype=float), **common))
    mouse_release_callbacks(viewer, read_only_mouse_event(
        type="mouse_release", position=tuple(position),
        pos=np.asarray(drag_to_pixel if drag_to_pixel is not None else (100.0, 100.0)),
        **common))


def by_node_ids(widget, role="inlet"):
    from haemolynx.gui.boundary_picking import method_setting

    rows_of(widget)[method_setting(role)].value = "node_ids"


def test_choosing_node_ids_offers_its_own_row_and_buttons(panel):
    widget, viewer, bc = panel
    by_node_ids(widget, "venule_boundary")

    # The box tools stay, to find the node to list.
    assert bc.state.actions["venule_boundary"] == {"pick_nodes", "clear_nodes"} | {"insert_box", "box_choice", "box_size", "box_step", "box_move", "box_nodes", "use_node", "remove_box"}
    assert rows_of(widget)["venule_boundary_node_ids"] in list(bc.holders["venule_boundary"])
    assert bc.actions["venule_boundary"].pick_nodes in list(bc.holders["venule_boundary"])


def test_picking_nodes_before_a_graph_exists_says_why(panel):
    widget, viewer, bc = panel
    by_node_ids(widget)

    bc.pick_nodes()

    assert "run at least '3. Graph'" in widget._haemolynx_report()
    assert bc.state.node_pick is None


def test_a_click_on_a_node_lists_it_and_marks_it(panel):
    from haemolynx.gui.boundary_picking import BC_NODE_IDS
    from haemolynx.gui.results import NODES

    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget)

    bc.pick_nodes()
    assert bc.state.node_pick == "inlet"
    assert bc.actions["inlet"].pick_nodes.text == "Stop picking nodes"
    assert viewer.layers[NODES].visible, "the nodes have to be seen to be clicked"
    assert viewer.layers.selection.active is viewer.layers[NODES], (
        "active, so the status bar names the node under the cursor"
    )

    click(viewer, Y_NODES[2])

    assert widget._haemolynx_values()["inlet_node_ids"] == [2]
    assert "Added node 2 to inlet" in widget._haemolynx_report()
    marked = viewer.layers[BC_NODE_IDS]
    assert np.allclose(marked.data, [Y_NODES[2]])
    assert list(marked.features["role"]) == ["inlet"]


def test_clicking_a_listed_node_again_takes_it_off(panel):
    from haemolynx.gui.boundary_picking import BC_NODE_IDS

    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget)
    bc.pick_nodes()

    click(viewer, Y_NODES[0])
    click(viewer, Y_NODES[3])
    click(viewer, Y_NODES[0])

    assert widget._haemolynx_values()["inlet_node_ids"] == [3]
    click(viewer, Y_NODES[3])
    assert widget._haemolynx_values()["inlet_node_ids"] == []
    assert BC_NODE_IDS not in viewer.layers, "nothing listed, nothing marked"


def test_a_junction_can_be_clicked_for_a_vessel_boundary(panel):
    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget, "arteriole_boundary")
    bc.role.value = "arteriole_boundary"

    bc.pick_nodes()
    click(viewer, Y_NODES[1])

    assert widget._haemolynx_values()["arteriole_boundary_node_ids"] == [1]


def test_a_drag_turns_the_view_and_picks_nothing(panel):
    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget)
    bc.pick_nodes()

    click(viewer, Y_NODES[2], drag_to_pixel=(160.0, 100.0))

    assert widget._haemolynx_values()["inlet_node_ids"] == []


def test_a_click_on_empty_space_says_so(panel):
    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget)
    bc.pick_nodes()

    click(viewer, (20.0, 200.0, 10.0))

    assert widget._haemolynx_values()["inlet_node_ids"] == []
    assert "No node under that click" in widget._haemolynx_report()


def test_pressing_the_button_again_gives_clicks_back_to_the_camera(panel):
    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget)

    bc.pick_nodes()
    assert bc.pick_node_click in viewer.mouse_drag_callbacks
    bc.pick_nodes()

    assert bc.state.node_pick is None
    assert bc.pick_node_click not in viewer.mouse_drag_callbacks
    assert bc.actions["inlet"].pick_nodes.text == "Pick nodes in the viewer"
    click(viewer, Y_NODES[2])
    assert widget._haemolynx_values()["inlet_node_ids"] == []


def test_moving_the_method_off_node_ids_stops_picking(panel):
    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget)
    bc.pick_nodes()

    rows_of(widget)["inlet_node_selection_method"].value = "coordinates"

    assert bc.state.node_pick is None
    assert bc.pick_node_click not in viewer.mouse_drag_callbacks


def test_clicks_follow_the_role_page_being_looked_at(panel):
    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget, "inlet")
    by_node_ids(widget, "outlet")
    bc.pick_nodes()

    bc.role.value = "outlet"
    click(viewer, Y_NODES[3])

    values = widget._haemolynx_values()
    assert values["outlet_node_ids"] == [3]
    assert values["inlet_node_ids"] == []
    assert bc.actions["outlet"].pick_nodes.text == "Stop picking nodes"
    assert bc.actions["inlet"].pick_nodes.text == "Pick nodes in the viewer"

    rows_of(widget)["venule_boundary_selection_method"].value = "edge_percent"
    bc.role.value = "venule_boundary"
    assert bc.state.node_pick is None, "a page that does not pick nodes takes none"


def test_clearing_a_roles_node_ids(panel):
    widget, viewer, bc = panel
    by_node_ids(widget)
    rows_of(widget)["inlet_node_ids"].value = [0, 2]

    bc.clear_nodes()

    assert widget._haemolynx_values()["inlet_node_ids"] == []


def test_typed_node_ids_are_marked_when_shown(panel):
    """Picking is one way in; the row is the other, and the layer follows it."""
    from haemolynx.gui.boundary_picking import BC_NODE_IDS

    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget, "outlet")
    rows_of(widget)["outlet_node_ids"].value = [3, 42]

    bc.show()

    assert np.allclose(viewer.layers[BC_NODE_IDS].data, [Y_NODES[3]])
    report = widget._haemolynx_report()
    assert "Not in the current graph: outlet [42]" in report


def test_node_ids_the_method_will_not_read_are_called_out(panel):
    widget, viewer, bc = panel
    rows_of(widget)["outlet_node_selection_method"].value = "edge_percent"
    rows_of(widget)["outlet_node_ids"].value = [3]

    bc.show()

    assert "1 outlet node ID(s) but outlet_node_selection_method is 'edge_percent'" in (
        widget._haemolynx_report()
    )


def test_a_near_miss_in_3d_takes_the_node_it_was_aimed_at(panel):
    """A dot a few microns across is a few pixels in a whole-volume 3D view;
    a click 5 um off it at zoom 1 (8 px of slack) still takes it, and one
    40 um away takes nothing."""
    from napari.utils._test_utils import read_only_mouse_event
    from napari.utils.interactions import mouse_press_callbacks, mouse_release_callbacks

    widget, viewer, bc = panel
    with_graph(viewer, bc)
    by_node_ids(widget)
    viewer.dims.ndisplay = 3
    viewer.camera.zoom = 1.0
    bc.pick_nodes()

    def click_3d(position):
        common = {"button": 1, "dims_displayed": [0, 1, 2],
                  "view_direction": [1.0, 0.0, 0.0], "pos": np.array([5.0, 5.0])}
        mouse_press_callbacks(viewer, read_only_mouse_event(
            type="mouse_press", position=tuple(position), **common))
        mouse_release_callbacks(viewer, read_only_mouse_event(
            type="mouse_release", position=tuple(position), **common))

    z, y, x = Y_NODES[2]
    click_3d((0.0, y + 5.0, x))
    assert widget._haemolynx_values()["inlet_node_ids"] == [2]

    click_3d((0.0, y + 40.0, x))
    assert widget._haemolynx_values()["inlet_node_ids"] == [2]
    assert "No node under that click" in widget._haemolynx_report()


# --- the volume method: insert a box, move it, choose a node in it ------------


def _box_panel(panel, method="volume"):
    """The panel with a small graph, automation off and the inlet on *method*.

    Open end 1 sits at (20, 50, 60); junction 2 just beside it; 3 far away.
    """
    import networkx as nx
    from types import SimpleNamespace

    widget, viewer, bc = panel
    no_bands(widget)
    rows_of(widget)["automated_vessel_assignment"].value = False
    rows_of(widget)["inlet_node_selection_method"].value = method
    graph = nx.MultiGraph()
    for node, pos in {1: (20, 50, 60), 2: (22, 52, 62), 3: (40, 150, 150),
                      4: (19, 47, 58)}.items():
        graph.add_node(node, pos=np.asarray(pos, dtype=float))
    graph.add_edges_from([(1, 2), (2, 3), (2, 4), (4, 3)])
    bc.state.results = SimpleNamespace(graph=graph)
    bc.role.value = "inlet"
    viewer.dims.ndisplay = 2
    viewer.dims.set_point(0, 20.0)
    viewer.camera.center = (20.0, 50.0, 60.0)
    return widget, viewer, bc


def _table_ids(bc, role="inlet"):
    table = bc.actions[role].box_nodes.table
    return [table.item(r, 0).text() for r in range(table.rowCount())]


def test_insert_a_box_puts_a_10um_box_in_the_middle_of_the_view(panel):
    from haemolynx.gui.boundary_picking import BC_BOX_NODES

    widget, viewer, bc = _box_panel(panel)
    bc.insert_box()

    assert rows_of(widget)["inlet_node_volumes"].value == [
        [[15.0, 45.0, 55.0], [25.0, 55.0, 65.0]]
    ]
    # Every node in the box is listed, the open end first.
    assert _table_ids(bc) == ["1", "2", "4"]
    assert bc.actions["inlet"].box_nodes.table.item(0, 2).text() == "open end"
    drawn = viewer.layers[BC_BOX_NODES]
    assert len(drawn.data) == 3
    assert drawn.size[0] > drawn.size[1], "the open end is drawn larger"
    # The region and its box are drawn like any other volume.
    assert regions_name("inlet") in viewer.layers and boxes_name("inlet") in viewer.layers


def test_box_size_and_arrows_edit_the_box(panel):
    widget, viewer, bc = _box_panel(panel)
    bc.insert_box()
    action = bc.actions["inlet"]

    action.box_size[2].value = 30.0          # x
    bc.flush_boxes()                          # what the settle timer does
    assert rows_of(widget)["inlet_node_volumes"].value == [
        [[15.0, 45.0, 45.0], [25.0, 55.0, 75.0]]
    ]
    action.box_step.value = 100.0
    action.box_move["right"].changed()
    bc.flush_boxes()
    assert rows_of(widget)["inlet_node_volumes"].value == [
        [[15.0, 45.0, 145.0], [25.0, 55.0, 175.0]]
    ]
    assert _table_ids(bc) == [], "moved off the vessel end"
    action.box_move["left"].changed()
    action.box_move["forward"].changed()      # through the slices: z
    bc.flush_boxes()
    assert rows_of(widget)["inlet_node_volumes"].value == [
        [[115.0, 45.0, 45.0], [125.0, 55.0, 75.0]]
    ]


def test_choosing_a_node_from_the_box_makes_it_the_inlet(panel):
    from haemolynx.gui.boundary_picking import BC_NODE_IDS

    widget, viewer, bc = _box_panel(panel)
    bc.insert_box()
    bc.actions["inlet"].box_nodes.table.selectRow(0)
    bc.actions["inlet"].use_node.changed()

    assert rows_of(widget)["inlet_node_ids"].value == [1]
    assert rows_of(widget)["inlet_node_selection_method"].value == "node_ids"
    assert BC_NODE_IDS in viewer.layers
    # The box tools stay, so another node can be found the same way.
    assert "insert_box" in bc.state.actions["inlet"]
    assert "now one of inlet's nodes" in widget._haemolynx_report()
    # Under node_ids the box is how the node was found, not a pick left unread.
    bc.show()
    assert "Not used" not in widget._haemolynx_report()


def test_use_node_without_a_selection_says_so(panel):
    widget, viewer, bc = _box_panel(panel)
    bc.insert_box()
    bc.use_node()
    assert rows_of(widget)["inlet_node_ids"].value == []
    assert rows_of(widget)["inlet_node_selection_method"].value == "volume"


def test_several_boxes_and_remove(panel):
    widget, viewer, bc = _box_panel(panel)
    bc.insert_box()
    viewer.camera.center = (20.0, 150.0, 150.0)
    bc.insert_box()
    action = bc.actions["inlet"]
    assert list(action.box_choice.choices) == ["Box 1", "Box 2"]
    assert action.box_choice.value == "Box 2"
    assert _table_ids(bc) == []                      # nothing near (150, 150)

    action.box_choice.value = "Box 1"
    assert _table_ids(bc) == ["1", "2", "4"]
    action.remove_box.changed()
    assert len(rows_of(widget)["inlet_node_volumes"].value) == 1
    assert rows_of(widget)["inlet_node_volumes"].value[0][0][1] == 145.0


def test_without_a_graph_the_box_list_says_to_run_graph_first(panel):
    widget, viewer, bc = _box_panel(panel)
    bc.state.results = None
    bc.insert_box()
    assert _table_ids(bc) == []
    assert "3. Graph" in bc.actions["inlet"].box_nodes.note.value


def test_box_nodes_go_when_the_role_leaves_the_box_method(panel):
    from haemolynx.gui.boundary_picking import BC_BOX_NODES

    widget, viewer, bc = _box_panel(panel)
    bc.insert_box()
    assert BC_BOX_NODES in viewer.layers
    rows_of(widget)["inlet_node_selection_method"].value = "coordinates"
    assert BC_BOX_NODES not in viewer.layers


def test_show_draws_no_inlet_outlet_band_while_the_masks_choose(panel):
    """The bands an unused edge_percent method describes are not what a run
    picks while the large-vessel masks choose the inlets and outlets; drawing
    them made the masks' choice look wrong. The run's own picks are shown."""
    from haemolynx.gui._widget import _apply_layers
    from haemolynx.gui.results import BOUNDARY_NODES, LayerSpec, StageLayers

    widget, viewer, bc = panel
    rows = rows_of(widget)
    rows["inlet_node_selection_method"].value = "edge_percent"
    rows["outlet_node_selection_method"].value = "edge_percent"
    rows["automated_vessel_assignment"].value = True
    rows["inlets_outlets_from_vessel_masks"].value = True
    _apply_layers(viewer, StageLayers(
        stage="assign_boundaries", title="Boundaries", note="",
        layers=(LayerSpec(
            kind="points", name=BOUNDARY_NODES,
            data=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            features={"role": np.asarray(["inlet", "inlet", "outlet"], dtype=object)},
            options={"visible": False},
        ),),
    ))

    bc.show()

    assert boxes_name("inlet") not in viewer.layers
    assert boxes_name("outlet") not in viewer.layers
    report = widget._haemolynx_report()
    assert "inlet band" not in report
    assert "2 inlet(s) and 1 outlet(s)" in report
    assert viewer.layers[BOUNDARY_NODES].visible

    # Choosing them by hand again draws the band that will be used.
    rows["inlets_outlets_from_vessel_masks"].value = False
    bc.show()
    assert boxes_name("inlet") in viewer.layers
    assert "chosen by the large-vessel masks" not in widget._haemolynx_report()


def test_box_lists_nodes_of_the_network_boundaries_runs_on(panel):
    """After a run the network on screen is the final one -- cut at the large
    vessels, pruned. A boundary node is chosen for the next Boundaries run, so
    the box lists the '3. Graph' network, where a large vessel's end still is."""
    import networkx as nx
    from types import SimpleNamespace

    widget, viewer, bc = _box_panel(panel)
    built = nx.MultiGraph()
    built.add_node(77, pos=np.asarray((20.0, 50.0, 60.0)))   # cut away later
    built.add_node(78, pos=np.asarray((40.0, 150.0, 150.0)))
    built.add_edge(77, 78)
    checkpoints = widget._haemolynx_checkpoints
    checkpoints.get = lambda stage: SimpleNamespace(graph=built) if stage == "build_network" else None

    bc.insert_box()

    assert _table_ids(bc) == ["77"]
    bc.actions["inlet"].box_nodes.table.selectRow(0)
    bc.use_node()
    assert rows_of(widget)["inlet_node_ids"].value == [77]


def test_clicking_the_arrows_moves_the_box_at_once_and_writes_it_once_settled(panel):
    """Each press only moves the role's own rectangle and 3D box; the setting,
    the node list and the box's nodes follow once the presses stop -- writing
    the row per press redrew every boundary layer twice, which lagged."""
    widget, viewer, bc = _box_panel(panel)
    bc.insert_box()
    bc.show()
    before = rows_of(widget)["inlet_node_volumes"].value
    action = bc.actions["inlet"]
    action.box_step.value = 100.0
    for _ in range(3):
        action.box_move["right"].changed()

    rectangle = np.asarray(viewer.layers[regions_name("inlet")].data[0])
    assert rectangle[:, 2].min() == pytest.approx(355.0), "drawn where it now is"
    assert rows_of(widget)["inlet_node_volumes"].value == before, "not written yet"

    bc.flush_boxes()
    assert rows_of(widget)["inlet_node_volumes"].value == [
        [[15.0, 45.0, 355.0], [25.0, 55.0, 365.0]]
    ]
    assert _table_ids(bc) == []
