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
    BC_COORDINATES,
    BC_REGIONS,
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


# --- what a config describes, drawn ------------------------------------------


def test_showing_a_config_puts_both_layers_in_the_viewer(panel):
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[10.0, 20.0, 30.0]]
    rows_of(widget)["output_node_volumes"].value = [A_BOX]

    bc.show()

    assert isinstance(viewer.layers[BC_COORDINATES], napari.layers.Points)
    assert isinstance(viewer.layers[BC_REGIONS], napari.layers.Shapes)
    assert tuple(viewer.layers[BC_COORDINATES].scale) == (1.0, 1.0, 1.0)


def test_a_picked_coordinate_lands_where_the_image_says_it_should(panel):
    """The claim the whole feature rests on: world coordinates are microns.

    The stack is 60 slices at 2 um, so its far z face is at 118 um. A
    coordinate there must sit at the far face of the image layer, not at slice
    118 -- which is off the end of a 60-slice stack.
    """
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[118.0, 0.0, 0.0]]

    bc.show()

    image_far_z = float(viewer.layers["stack"].extent.world[1][0])
    assert viewer.layers[BC_COORDINATES].data[0][0] == pytest.approx(image_far_z)


def test_the_region_is_drawn_as_a_rectangle_at_the_boxs_centre(panel):
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]

    bc.show()

    corners = viewer.layers[BC_REGIONS].data[0]
    assert corners.shape == (4, 3)
    assert corners[:, 0].tolist() == [50.0] * 4, "planar, at the box's z centre"
    assert corners[:, 1].min() == pytest.approx(100.0)
    assert corners[:, 2].max() == pytest.approx(150.0)


def test_showing_twice_updates_rather_than_duplicates(panel):
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[1.0, 2.0, 3.0]]
    bc.show()
    same = viewer.layers[BC_COORDINATES]

    rows_of(widget)["starting_node_coordinates"].value = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
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
    rows_of(widget)["output_node_volumes"].value = [A_BOX]
    bc.show()

    _clear_our_layers(viewer)

    assert BC_COORDINATES not in viewer.layers
    assert BC_REGIONS not in viewer.layers


# --- the colours, which were wrong before ------------------------------------


def test_each_role_gets_its_own_colour_whatever_order_they_appear_in(panel):
    """A layer holding only output nodes used to draw them in starting's blue.

    The cycle was handed to napari as a bare list of colours, which it pairs
    with the values in the order it first encounters them, not by the labels
    they were declared against.
    """
    widget, viewer, bc = panel
    rows_of(widget)["output_node_coordinates"].value = [[1.0, 1.0, 1.0]]
    rows_of(widget)["starting_node_coordinates"].value = [[2.0, 2.0, 2.0]]

    bc.show()

    expected = dict(role_colours())
    layer = viewer.layers[BC_COORDINATES]
    for role, colour in zip(layer.features["role"], layer.face_color):
        assert np.allclose(colour, expected[role], atol=0.01), role


def test_the_colours_survive_a_second_show(panel):
    """`options` are applied only on first add, so colour has to come through
    the colour path or it goes stale the moment a layer is updated."""
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[1.0, 1.0, 1.0]]
    bc.show()
    rows_of(widget)["output_node_coordinates"].value = [[2.0, 2.0, 2.0]]

    bc.show()

    expected = dict(role_colours())
    layer = viewer.layers[BC_COORDINATES]
    for role, colour in zip(layer.features["role"], layer.face_color):
        assert np.allclose(colour, expected[role], atol=0.01), role


# --- editing a layer edits the settings --------------------------------------


def test_adding_a_point_writes_it_into_the_chosen_roles_setting(panel):
    widget, viewer, bc = panel
    bc.role.value = "output"
    bc.pick()

    layer = viewer.layers[BC_COORDINATES]
    layer.data = np.array([[11.0, 22.0, 33.0]])

    assert widget._haemolynx_values()["output_node_coordinates"] == [[11.0, 22.0, 33.0]]
    assert widget._haemolynx_values()["starting_node_coordinates"] == []


def test_the_row_can_read_back_what_was_written_into_it(panel):
    """magicgui stores `str(value)` and parses it with `literal_eval`, so a
    numpy float reaching the row would break it the next time it is touched."""
    widget, viewer, bc = panel
    bc.pick()
    viewer.layers[BC_COORDINATES].data = np.array([[1.5, 2.5, 3.5]])

    value = rows_of(widget)["starting_node_coordinates"].value
    assert all(type(v) is float for v in value[0]), "a numpy float breaks the row"
    assert ast.literal_eval(str(value)) == [[1.5, 2.5, 3.5]]


def test_deleting_a_point_removes_exactly_that_coordinate(panel):
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [
        [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]
    ]
    bc.show()

    layer = viewer.layers[BC_COORDINATES]
    layer.selected_data = {1}
    layer.remove_selected()

    assert widget._haemolynx_values()["starting_node_coordinates"] == [
        [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]
    ]


def test_moving_a_point_updates_it_in_place(panel):
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[1.0, 1.0, 1.0]]
    bc.show()

    layer = viewer.layers[BC_COORDINATES]
    layer.data = np.array([[7.0, 8.0, 9.0]])

    assert widget._haemolynx_values()["starting_node_coordinates"] == [[7.0, 8.0, 9.0]]


def test_syncing_the_same_edit_twice_changes_nothing(panel):
    """`events.data` fires twice per edit, so the handler has to be idempotent."""
    widget, viewer, bc = panel
    bc.pick()
    viewer.layers[BC_COORDINATES].data = np.array([[1.0, 2.0, 3.0]])
    once = widget._haemolynx_values()["starting_node_coordinates"]

    bc.sync()
    bc.sync()

    assert widget._haemolynx_values()["starting_node_coordinates"] == once


def test_editing_points_does_not_wipe_the_regions(panel):
    """The two layers share a sync; one must not clear the other's settings."""
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]
    bc.show()

    viewer.layers[BC_COORDINATES].data = np.array([[1.0, 2.0, 3.0]])

    assert widget._haemolynx_values()["output_node_volumes"] == [A_BOX]


def test_a_region_edited_in_the_viewer_reaches_the_setting(panel):
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]
    bc.show()

    layer = viewer.layers[BC_REGIONS]
    corners, _ = rectangle_from_box([0.0, 10.0, 20.0], [100.0, 60.0, 90.0])
    layer.data = [corners]

    lo, hi = widget._haemolynx_values()["output_node_volumes"][0]
    assert lo[1:] == [10.0, 20.0] and hi[1:] == [60.0, 90.0]


def test_the_depth_slider_resizes_the_selected_region(panel):
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]
    bc.show()
    viewer.layers[BC_REGIONS].selected_data = {0}

    bc.depth.value = 20.0

    lo, hi = widget._haemolynx_values()["output_node_volumes"][0]
    assert hi[0] - lo[0] == pytest.approx(20.0)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(50.0), "still centred where it was"


def test_clearing_a_roles_regions_leaves_the_other_roles_alone(panel):
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]
    rows_of(widget)["starting_node_volumes"].value = [A_BOX]
    bc.show()

    bc.role.value = "output"
    bc.clear()

    assert widget._haemolynx_values()["output_node_volumes"] == []
    assert widget._haemolynx_values()["starting_node_volumes"] == [A_BOX]


def test_assigning_a_selected_point_to_another_role_moves_it_between_settings(panel):
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[1.0, 2.0, 3.0]]
    bc.show()

    viewer.layers[BC_COORDINATES].selected_data = {0}
    bc.role.value = "venule_boundary"
    bc.assign()

    values = widget._haemolynx_values()
    assert values["starting_node_coordinates"] == []
    assert values["venule_boundary_node_coordinates"] == [[1.0, 2.0, 3.0]]


# --- what cannot be done, said rather than silently failing ------------------


def test_drawing_a_region_is_refused_in_the_3d_view(panel):
    """napari forces a Shapes layer out of edit mode when ndisplay is 3."""
    widget, viewer, bc = panel
    viewer.dims.ndisplay = 3

    bc.draw()

    assert "2D" in widget._haemolynx_report()
    assert BC_REGIONS not in viewer.layers


def test_snapping_before_a_run_says_why_rather_than_doing_nothing(panel):
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[1.0, 2.0, 3.0]]
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

    rows_of(widget)["starting_node_coordinates"].value = [[48.0, 61.0, 69.0]]
    bc.show()
    bc.snap()

    assert widget._haemolynx_values()["starting_node_coordinates"] == [[50.0, 60.0, 70.0]]
    assert "um" in widget._haemolynx_report()


def test_picks_that_the_run_will_not_read_are_called_out(panel):
    """A role only reads its coordinates when its method says so."""
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_selection_method"].value = "edge_percent"
    rows_of(widget)["starting_node_coordinates"].value = [[1.0, 2.0, 3.0]]

    bc.show()

    assert "Not used" in widget._haemolynx_report()
    assert "edge_percent" in widget._haemolynx_report()


def test_a_config_that_cannot_be_read_is_reported_not_raised(panel):
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[1.0, 2.0]]

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
    for role_setting in ("starting_node_coordinates", "output_node_volumes"):
        assert values[role_setting] == schema[role_setting].default


def test_the_controls_sit_on_the_boundaries_tab(panel):
    widget, viewer, bc = panel
    assert bc.widget is not None
    assert [str(choice) for choice in bc.role.choices][0] == "starting"


# --- a region reads as the volume it is --------------------------------------


def test_a_region_is_drawn_as_a_box_not_a_flat_rectangle(panel):
    """A rectangle on one slice says nothing about how deep the box goes."""
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]

    bc.show()

    regions = viewer.layers[BC_REGIONS]
    assert list(regions.shape_type).count("rectangle") == 1
    assert list(regions.shape_type).count("line") == 12
    drawn = np.concatenate([np.asarray(s) for s in regions.data], axis=0)
    assert drawn[:, 0].min() == pytest.approx(A_BOX[0][0])
    assert drawn[:, 0].max() == pytest.approx(A_BOX[1][0])


def test_the_outline_survives_a_second_show(panel):
    """Growing a Shapes layer through `.data` turns every shape into a polygon.

    The box would quietly flatten back into thirteen polygons on the second
    draw, so `shape_type` has to be re-applied on update, not only on add.
    """
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]
    bc.show()

    rows_of(widget)["starting_node_volumes"].value = [[[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]]
    bc.show()

    regions = viewer.layers[BC_REGIONS]
    assert len(regions.data) == 26
    assert list(regions.shape_type).count("line") == 24
    assert list(regions.shape_type).count("rectangle") == 2


def test_reading_the_regions_back_counts_boxes_not_segments(panel):
    """Thirteen shapes are one region; a sync that missed that would multiply."""
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]
    bc.show()

    bc.sync()

    assert len(widget._haemolynx_values()["output_node_volumes"]) == 1


# --- the layers follow the form ----------------------------------------------


def test_editing_a_setting_redraws_the_layers(panel):
    """The form is one of the two authoritative directions, so it has to show."""
    widget, viewer, bc = panel
    bc.show()
    assert len(viewer.layers[BC_COORDINATES].data) == 0

    rows_of(widget)["starting_node_coordinates"].value = [[10.0, 20.0, 30.0]]

    assert len(viewer.layers[BC_COORDINATES].data) == 1
    assert viewer.layers[BC_COORDINATES].data[0] == pytest.approx([10.0, 20.0, 30.0])


def test_changing_a_method_updates_what_the_panel_says_is_used(panel):
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]
    rows_of(widget)["output_node_selection_method"].value = "volume"
    bc.show()
    assert "Not used" not in widget._haemolynx_report()

    rows_of(widget)["output_node_selection_method"].value = "coordinates"

    assert "Not used" in widget._haemolynx_report()
    assert BC_REGIONS in viewer.layers, "a configured region is never silently dropped"


def test_a_region_drawn_after_a_settings_edit_still_arrives(panel):
    """The redraw must leave the layer editable, not replace it with a picture."""
    widget, viewer, bc = panel
    bc.draw()
    rows_of(widget)["output_node_volumes"].value = [A_BOX]

    regions = viewer.layers[BC_REGIONS]
    assert regions.mode != "pan_zoom"


# --- only the settings the chosen method reads ------------------------------


def test_only_the_settings_the_chosen_method_reads_are_shown(panel):
    widget, viewer, bc = panel
    rows_of(widget)["output_node_selection_method"].value = "coordinates"
    rows_of(widget)["starting_node_selection_method"].value = "volume"

    visible = bc.state.visible
    assert "output_node_coordinates" in visible
    assert "starting_node_volumes" in visible
    assert "output_node_volumes" not in visible
    assert "starting_node_coordinates" not in visible
    # `.visible` on a row of a tab that is not on screen reads False whatever
    # was set, so what the panel hid is recorded rather than queried.
    assert "output_node_volumes" in bc.state.hidden
    assert "output_node_coordinates" not in bc.state.hidden


def test_a_method_row_is_never_hidden(panel):
    """It is the row that decides which of the others you need."""
    widget, viewer, bc = panel
    # `all_degree_1` takes every terminal, so it configures nothing at all --
    # the case where hiding everything would hide the way back.
    rows_of(widget)["output_node_selection_method"].value = "all_degree_1"

    assert "output_node_selection_method" not in bc.state.hidden
    assert "output_node_coordinates" in bc.state.hidden


def test_editing_one_role_does_not_wipe_another(panel):
    """Applying a layer fires its own data event.

    Unguarded, that event reads the half-written layer straight back into the
    settings, and the role whose features had not been written yet loses its
    boxes -- silently, since the row it emptied is the one you were not
    looking at.
    """
    widget, viewer, bc = panel
    rows_of(widget)["output_node_volumes"].value = [A_BOX]
    bc.show()

    rows_of(widget)["starting_node_volumes"].value = [[[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]]

    values = widget._haemolynx_values()
    assert len(values["output_node_volumes"]) == 1
    assert len(values["starting_node_volumes"]) == 1
    assert len(viewer.layers[BC_REGIONS].data) == 26


def test_a_setting_sits_below_the_method_that_asks_for_it(panel):
    widget, viewer, bc = panel
    names = list(rows_of(widget))
    order = bc.row_order([n for n in names if n.startswith(("starting_", "output_"))])

    assert order.index("output_node_selection_method") < order.index("output_node_coordinates")
    assert order.index("starting_node_volumes") < order.index("output_node_selection_method")


def test_the_ordering_keeps_every_row_it_was_given(panel):
    """A row this module does not know about must not disappear off the tab."""
    widget, viewer, bc = panel
    names = list(rows_of(widget))

    assert sorted(bc.row_order(names)) == sorted(names)


# --- one sub-tab per role ----------------------------------------------------


def test_there_is_one_sub_tab_per_role(panel):
    widget, viewer, bc = panel
    tabs = bc.state.tabs

    assert tabs.count() == 4
    assert [tabs.tabText(i) for i in range(4)] == [
        "Starting", "Output", "Arteriole", "Venule"
    ]


def test_choosing_a_sub_tab_chooses_the_role(panel):
    """The tab is the role: two ways to say it could disagree."""
    widget, viewer, bc = panel

    bc.state.tabs.setCurrentIndex(1)

    assert str(bc.role.value) == "output"


def test_choosing_the_role_moves_the_sub_tab(panel):
    widget, viewer, bc = panel

    bc.role.value = "venule_boundary"

    assert bc.state.tabs.currentIndex() == 3


def test_a_role_only_shows_its_own_settings(panel):
    """The complaint this answers: four near-identical methods on one page."""
    widget, viewer, bc = panel
    starting_page = bc.state.tabs.widget(0)
    natives = set(starting_page.findChildren(type(rows_of(widget)[
        "starting_node_selection_method"].native)))

    assert rows_of(widget)["starting_node_selection_method"].native in natives
    assert rows_of(widget)["output_node_selection_method"].native not in natives


def test_a_picked_point_takes_the_role_of_the_open_sub_tab(panel):
    widget, viewer, bc = panel
    bc.show()
    bc.state.tabs.setCurrentIndex(1)

    points = viewer.layers[BC_COORDINATES]
    points.add([[7.0, 8.0, 9.0]])

    assert widget._haemolynx_values()["output_node_coordinates"] == [[7.0, 8.0, 9.0]]
    assert widget._haemolynx_values()["starting_node_coordinates"] == []


def test_the_shared_band_settings_are_not_on_a_role_page(panel):
    """One axis and one pair of bands describe the whole network."""
    widget, viewer, bc = panel
    axis = rows_of(widget)["boundary_axis"].native

    for index in range(bc.state.tabs.count()):
        assert axis not in bc.state.tabs.widget(index).findChildren(type(axis))


# --- the depth slider on a real, anisotropic stack ---------------------------


def test_showing_survives_a_stack_whose_depth_the_slider_rounds(panel):
    """`FloatSlider` rounds the maximum it is given and raises on a value past
    it, so defaulting the depth to the stack's exact span blew up on any stack
    whose depth was not a round number -- which is any anisotropic one."""
    widget, viewer, bc = panel

    bc.show()

    span = float(viewer.layers["stack"].extent.world[1][0]
                 - viewer.layers["stack"].extent.world[0][0])
    assert bc.depth.value == pytest.approx(span, abs=1e-3)
    assert bc.depth.value <= bc.depth.max


# --- coordinates that do not lie on the image --------------------------------


def test_a_coordinate_off_the_image_is_reported_with_the_images_size(panel):
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[5000.0, 5.0, 5.0]]

    bc.show()

    report = widget._haemolynx_report()
    assert "Outside the image" in report
    assert "microns, not voxel indices" in report


def test_a_coordinate_on_the_image_is_not_reported(panel):
    widget, viewer, bc = panel
    rows_of(widget)["starting_node_coordinates"].value = [[10.0, 20.0, 30.0]]

    bc.show()

    assert "Outside the image" not in widget._haemolynx_report()
