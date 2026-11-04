"""Building up a list of perturbations in a real panel.

`test_gui_perturbation_editing.py` decides what the editor should do without a
display. What is left for here is what only a built panel answers: that the tab
exists and has something on it, that choosing a type reveals that type's
options, that "+" really does add another dropdown -- and the one that matters
most, that an editor writes into the perturbation and **not** into the flat
settings the baseline run reads.

That last one is the whole reason a perturbation is a nested entry rather than
a set of extra rows. `arteriole_diameter_change_percent` is a typed override,
and the panel sends every ordinary row to the run, so an editor that wrote
into `rows` would move the baseline that every perturbation is differenced
against.

They need napari, a Qt binding and a display, so they are marked `gui` and
skipped everywhere those are missing. CI runs them on 3.11 under xvfb.
"""
from __future__ import annotations

import ast

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.perturbation_editing import (  # noqa: E402
    PERTURBATION_TYPES,
    UNCHOSEN,
    perturbation_type_choices,
    rows_for_type,
)
from haemolynx.pipeline.progress import STAGES  # noqa: E402

pytestmark = pytest.mark.gui

#: The tab these controls own, taken from the stage rather than typed out, so
#: renaming the stage cannot leave this looking for a tab that is not there.
TAB_TITLE = next(
    stage.title for stage in STAGES if stage.call == "run_perturbations"
)

A_TYPE = "arteriole_diameter_change"
A_PERCENT = "arteriole_diameter_change_percent"
COMBINED_TYPE = "arteriole_and_pericyte_diameter_change"


@pytest.fixture
def panel(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((8, 8, 8)), name="stack")
    widget = settings_widget(napari_viewer=viewer)
    return widget, viewer, widget._haemolynx_perturbations


def rows_of(widget):
    return widget._haemolynx_rows()


def tab_page(widget, title):
    """The widget behind one tab, whatever it was wrapped in."""
    from qtpy.QtWidgets import QScrollArea, QTabWidget

    tabs = widget.findChild(QTabWidget)
    for index in range(tabs.count()):
        if tabs.tabText(index) == title:
            page = tabs.widget(index)
            return page.widget() if isinstance(page, QScrollArea) else page
    raise AssertionError(f"no tab called {title!r}")


# --- it is there and it builds ----------------------------------------------


def test_the_controls_are_built(panel):
    _widget, _viewer, perturbations = panel
    assert perturbations is not None


def test_the_panel_builds_without_a_viewer():
    """No viewer, no controls -- and still a panel."""
    widget = settings_widget(napari_viewer=None)
    assert widget is not None
    assert widget._haemolynx_perturbations is None


def test_the_perturbations_tab_has_something_on_it(panel):
    """A tab whose page laid nothing out would build, and be blank."""
    widget, _viewer, _perturbations = panel
    page = tab_page(widget, TAB_TITLE)
    assert page.isAncestorOf(rows_of(widget)["perturbations"].native)
    assert page.isAncestorOf(rows_of(widget)["run_perturbations"].native)


def test_the_controls_sit_on_the_perturbations_tab(panel):
    """The "+" button is on that tab and not somewhere else in the panel."""
    widget, _viewer, perturbations = panel
    page = tab_page(widget, TAB_TITLE)
    assert page.isAncestorOf(perturbations.add_button.native)
    assert page.isAncestorOf(perturbations.holder.native)


def test_building_the_controls_leaves_the_list_empty(panel):
    """A panel nobody has touched configures no perturbations."""
    widget, _viewer, perturbations = panel
    assert perturbations.entries() == []
    assert widget._haemolynx_values()["perturbations"] == []


# --- one dropdown per perturbation ------------------------------------------


def test_adding_one_gives_one_dropdown(panel):
    _widget, _viewer, perturbations = panel

    perturbations.add()

    assert len(perturbations.editors()) == 1
    choices = list(perturbations.editors()[0].type.choices)
    values = [c[1] if isinstance(c, (tuple, list)) else c for c in choices]
    assert values == list(PERTURBATION_TYPES)
    assert perturbations.editors()[0].type.value == UNCHOSEN
    assert any(
        "constriction/dilation" in label
        for label, _value in perturbation_type_choices()
    )


def test_perturbation_controls_carry_descriptive_tooltips(panel):
    """Hover text on name, type, Add/Remove, and each setting row."""
    from haemolynx.gui.form import field_for
    from haemolynx.gui.perturbation_editing import (
        ADD_TOOLTIP,
        EDITOR_SETTINGS,
        NAME_TOOLTIP,
        REMOVE_TOOLTIP,
        TYPE_TOOLTIP,
    )
    from haemolynx.pipeline import default_schema

    widget, _viewer, perturbations = panel
    schema = default_schema()
    perturbations.add()
    perturbations.choose_type(0, A_TYPE)

    assert perturbations.add_button.tooltip == ADD_TOOLTIP
    editor = perturbations.editors()[0]
    assert editor.name.tooltip == NAME_TOOLTIP
    assert editor.type.tooltip == TYPE_TOOLTIP
    assert editor.remove.tooltip == REMOVE_TOOLTIP
    for name in EDITOR_SETTINGS:
        expected = field_for(schema[name]).help
        assert editor.editors[name].tooltip == expected, name
    assert A_PERCENT in editor.shown
    assert "constriction" in editor.editors[A_PERCENT].tooltip.lower()
    # Keep the flat settings row wired too (always-visible tab).
    assert rows_of(widget)["run_perturbations"].tooltip


def test_adding_twice_gives_a_second_identical_dropdown(panel):
    """Repeatable: the user asked for N perturbations, not two."""
    _widget, _viewer, perturbations = panel

    perturbations.add()
    perturbations.add()

    assert len(perturbations.editors()) == 2
    assert len(perturbations.entries()) == 2
    names = [editor.name.value for editor in perturbations.editors()]
    assert len(set(names)) == 2, "each names its own output directory"


def test_a_new_one_runs_nothing_until_it_is_told_to(panel):
    _widget, _viewer, perturbations = panel

    perturbations.add()

    assert perturbations.entries()[0]["type"] == UNCHOSEN
    assert perturbations.entries()[0]["overrides"] == {}


def test_removing_one_takes_its_dropdown_with_it(panel):
    _widget, _viewer, perturbations = panel
    perturbations.add()
    perturbations.add()

    perturbations.remove(0)

    assert len(perturbations.editors()) == 1
    assert len(perturbations.entries()) == 1


def test_removing_the_first_leaves_the_second_editable(panel):
    """The editors are rebuilt because removing one moves the rest's index."""
    _widget, _viewer, perturbations = panel
    perturbations.add()
    perturbations.add()
    perturbations.choose_type(1, A_TYPE)
    kept = perturbations.entries()[1]["name"]

    perturbations.remove(0)
    perturbations.editors()[0].editors[A_PERCENT].value = 50

    assert perturbations.entries() == [
        {"name": kept, "type": A_TYPE, "overrides": {A_PERCENT: 50}}
    ]


# --- choosing a type reveals that type's options ----------------------------


def test_choosing_a_type_reveals_its_options(panel):
    _widget, _viewer, perturbations = panel
    perturbations.add()

    perturbations.choose_type(0, A_TYPE)

    editor = perturbations.editors()[0]
    assert editor.shown == set(rows_for_type(A_TYPE))
    assert A_PERCENT in editor.shown


def test_combined_type_shows_arteriole_percent_first(panel):
    """Arteriole % is a first-class knob on the combined type, not buried."""
    _widget, _viewer, perturbations = panel
    perturbations.add()

    perturbations.choose_type(0, COMBINED_TYPE)

    editor = perturbations.editors()[0]
    assert A_PERCENT in editor.shown
    assert A_PERCENT in editor.editors
    assert list(rows_for_type(COMBINED_TYPE))[0] == A_PERCENT
    assert editor.layout_order[0] == A_PERCENT
    assert "constriction" in editor.editors[A_PERCENT].label.lower()
    assert "%" in editor.editors[A_PERCENT].label


def test_choosing_a_type_hides_the_other_types_options(panel):
    """Hidden, not greyed out: a row you cannot use is a row you do not want
    to read past."""
    _widget, _viewer, perturbations = panel
    perturbations.add()

    perturbations.choose_type(0, A_TYPE)

    editor = perturbations.editors()[0]
    assert "inlet_pressure_min_pa" in editor.hidden
    assert editor.shown.isdisjoint(editor.hidden)
    # `.visible` reads False for anything on a tab that is not on screen, so
    # only the hidden case can be asked of the widget itself.
    assert editor.editors["inlet_pressure_min_pa"].visible is False


def test_an_unchosen_entry_shows_no_options_at_all(panel):
    _widget, _viewer, perturbations = panel

    perturbations.add()

    assert perturbations.editors()[0].shown == set()


def test_changing_a_type_reveals_the_new_ones_options(panel):
    _widget, _viewer, perturbations = panel
    perturbations.add()
    perturbations.choose_type(0, A_TYPE)

    perturbations.choose_type(0, "pressure_and_pericyte_sweep")

    editor = perturbations.editors()[0]
    assert editor.shown == set(rows_for_type("pressure_and_pericyte_sweep"))
    assert A_PERCENT in editor.hidden


def test_each_entry_reveals_its_own_type_independently(panel):
    _widget, _viewer, perturbations = panel
    perturbations.add()
    perturbations.add()

    perturbations.choose_type(0, A_TYPE)
    perturbations.choose_type(1, "pericyte_dilation_sweep")

    assert A_PERCENT in perturbations.editors()[0].shown
    assert A_PERCENT in perturbations.editors()[1].hidden


# --- an editor writes into the perturbation, not into the settings ----------


def test_an_editor_writes_into_that_perturbations_overrides(panel):
    _widget, _viewer, perturbations = panel
    perturbations.add()
    perturbations.choose_type(0, A_TYPE)

    perturbations.editors()[0].editors[A_PERCENT].value = 40

    assert perturbations.entries()[0]["overrides"][A_PERCENT] == pytest.approx(40)


def test_an_editor_does_not_touch_the_flat_settings(panel):
    """The guarantee the whole design rests on: the baseline does not move.

    `arteriole_diameter_change_percent` is claimed by the Perturbations tab so
    a Field exists to clone, and `current_values()` reads every claimed row.
    If the editor wrote there, every perturbation would be measured against a
    baseline it had already changed.
    """
    widget, _viewer, perturbations = panel
    before = widget._haemolynx_values()[A_PERCENT]
    perturbations.add()
    perturbations.choose_type(0, A_TYPE)

    perturbations.editors()[0].editors[A_PERCENT].value = 40

    assert widget._haemolynx_values()[A_PERCENT] == before
    assert perturbations.entries()[0]["overrides"][A_PERCENT] == pytest.approx(40)


def test_no_editor_appears_in_what_the_panel_would_run(panel):
    """The editors reach a run through the one `perturbations` value."""
    widget, _viewer, perturbations = panel
    keys_before = set(widget._haemolynx_values())
    perturbations.add()
    perturbations.choose_type(0, "pericyte_diameter_change")

    assert set(widget._haemolynx_values()) == keys_before


def test_a_new_entrys_options_start_at_what_the_run_configured(panel):
    """A config's own constriction length (a row on no tab) is what a new
    pericyte entry starts from. The editors started at the schema's 40 um,
    and choosing the type wrote that over the config's 20 um."""
    widget, _viewer, perturbations = panel
    rows_of(widget)["constriction_length_um"].value = 20.0
    perturbations.add()

    perturbations.choose_type(0, "pericyte_diameter_change")

    assert perturbations.entries()[0]["overrides"]["constriction_length_um"] == pytest.approx(20.0)
    assert widget._haemolynx_values()["constriction_length_um"] == pytest.approx(20.0)


def test_listed_vessels_are_written_with_the_ends_of_the_vessels_shown(panel):
    """A branchID typed from the hover is written with the vessel it named
    on the network shown, so a network built again since is caught."""
    from haemolynx.gui._widget import _apply_layers
    from haemolynx.gui.results import VESSELS
    from test_gui_results_widget import a_run

    widget, viewer, perturbations = panel
    for group in a_run():
        _apply_layers(viewer, group)
    features = viewer.layers[VESSELS].features
    ends = {
        int(i): [int(u), int(v), int(k)]
        for i, u, v, k in zip(features["edge_index"], features["u"], features["v"], features["key"])
    }
    ids = sorted(ends)
    assert ids
    perturbations.add()
    perturbations.choose_type(0, "capillary_block")
    editors = perturbations.editors()[0].editors
    editors["capillary_block_selection"].value = "vessel_ids"
    editors["capillary_block_vessel_ids"].value = ids

    written = perturbations.entries()[0]["overrides"]["capillary_block_vessel_ids"]
    assert written == [[index, *ends[index]] for index in ids]


def test_what_the_editors_say_reaches_the_run_through_the_one_row(panel):
    widget, _viewer, perturbations = panel
    perturbations.add()
    perturbations.choose_type(0, A_TYPE)
    perturbations.editors()[0].editors[A_PERCENT].value = 40

    configured = widget._haemolynx_values()["perturbations"]

    assert [entry["type"] for entry in configured] == [A_TYPE]
    assert configured[0]["overrides"][A_PERCENT] == pytest.approx(40)


def test_the_row_can_read_back_what_the_editors_put_in_it(panel):
    """The row is a LiteralEvalLineEdit: it stores `str(value)` and parses it
    back, so a numpy scalar or a Path in there is a value it cannot read."""
    widget, _viewer, perturbations = panel
    perturbations.add()
    perturbations.choose_type(0, A_TYPE)
    perturbations.editors()[0].editors[A_PERCENT].value = 40

    text = rows_of(widget)["perturbations"].native.text()

    assert ast.literal_eval(text) == rows_of(widget)["perturbations"].value


# --- the row and the editors are two views of one list ----------------------


def test_editing_the_row_by_hand_rebuilds_the_editors(panel):
    """A config loaded into the row has to show up as editors."""
    widget, _viewer, perturbations = panel

    rows_of(widget)["perturbations"].value = [
        {"name": "art_dilate_20", "type": A_TYPE, "overrides": {A_PERCENT: 20}}
    ]

    assert len(perturbations.editors()) == 1
    editor = perturbations.editors()[0]
    assert editor.type.value == A_TYPE
    assert editor.name.value == "art_dilate_20"
    assert editor.editors[A_PERCENT].value == pytest.approx(20)


def test_a_config_with_a_bad_entry_still_opens(panel):
    """Reading is lenient so that the panel can show what is wrong."""
    widget, _viewer, perturbations = panel

    rows_of(widget)["perturbations"].value = [{"name": "x", "type": "nonsense"}]

    assert len(perturbations.editors()) == 1
    assert perturbations.editors()[0].type.value == UNCHOSEN


def test_renaming_an_entry_reaches_the_setting(panel):
    widget, _viewer, perturbations = panel
    perturbations.add()

    perturbations.editors()[0].name.value = "art_dilate_20"

    assert widget._haemolynx_values()["perturbations"][0]["name"] == "art_dilate_20"


def test_a_name_that_could_not_be_a_directory_is_reported(panel):
    """The name is the output directory, so a separator in it is a problem."""
    widget, _viewer, perturbations = panel
    perturbations.add()

    perturbations.editors()[0].name.value = "../elsewhere"

    assert "directory" in widget._haemolynx_report()


# --- orphaned Perturbations rows must not float at startup -------------------


#: Flat ``rows[]`` widgets claimed by Perturbations but never parented there.
#: Editors clone their own copies; Diameters hide-when-unmet must not ``show()``
#: these unparented shells (``147a545`` + ``30c605e`` floating-window class).
_FLOATER_REGRESSION_SETTINGS = (
    "do_pericyte_construction",
    "use_pericyte_mask_constriction",
    "use_probabilistic_pericyte_constriction",
    "run_pericyte_resistance_comparison",
)


def test_orphaned_pericyte_rows_are_not_top_level_windows_at_startup(
    make_napari_viewer,
):
    """Snapshot top-levels *before* construction; these four must not float.

    ``30c605e`` / ``32645b9`` retabbed them onto Perturbations without parenting
    the ``rows[]`` shells. ``147a545`` put Diameters in
    ``HIDE_WHEN_UNMET_SECTIONS``, so ``apply_prerequisites`` set ``visible=True``
    under default ``run_haemodynamics`` and each became a top-level window.
    """
    from qtpy.QtWidgets import QApplication

    viewer = make_napari_viewer()
    before = {id(w) for w in QApplication.topLevelWidgets() if w.isVisible()}

    widget = settings_widget(napari_viewer=viewer)
    widget.show()
    QApplication.processEvents()

    rows = widget._haemolynx_rows()
    for name in _FLOATER_REGRESSION_SETTINGS:
        native = rows[name].native
        assert rows[name].visible is False, name
        assert native.parent() is not None, name
        assert not native.isWindow(), name

    appeared = [
        w
        for w in QApplication.topLevelWidgets()
        if w.isVisible() and id(w) not in before
    ]
    for name in _FLOATER_REGRESSION_SETTINGS:
        native = rows[name].native
        assert native not in appeared, name
        label = name.replace("_", " ")
        titles = [
            (w.windowTitle() or w.objectName() or type(w).__name__) for w in appeared
        ]
        assert not any(label.lower() in str(title).lower() for title in titles), (
            f"{name} floated as {titles}"
        )

    if "run_haemodynamics" in rows:
        rows["run_haemodynamics"].value = False
        QApplication.processEvents()
        rows["run_haemodynamics"].value = True
        QApplication.processEvents()
        for name in _FLOATER_REGRESSION_SETTINGS:
            assert rows[name].visible is False, name
            assert not rows[name].native.isWindow(), name


# --- saved for later runs -----------------------------------------------------


def _saved_panel(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    return widget, widget._haemolynx_perturbations


def _arteriole_entry(perturbations, name="art_dilate_20", percent=20):
    perturbations.add()
    index = len(perturbations.editors()) - 1
    perturbations.editors()[index].name.value = name
    perturbations.choose_type(index, A_TYPE)
    perturbations.editors()[index].editors[A_PERCENT].value = percent
    return index


def _choose_saved(perturbations, key):
    """Pick *key* in the saved-perturbation menu as a user does."""
    menu = perturbations.saved_menu
    index = menu.findData(key)
    assert index > 0, key
    menu.activated.emit(index)


def test_a_perturbation_saved_in_one_panel_is_added_to_the_list_in_another(
    make_napari_viewer, monkeypatch, tmp_path
):
    """Saved under its name with its options; in a later panel the menu lists
    it, and choosing it adds it to the end of the list -- its options in its
    editor, the baseline rows untouched."""
    from haemolynx.pipeline.saved_perturbations import PERTURBATION_DIR_ENV, read_saved_perturbation

    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(tmp_path))
    first, perturbations = _saved_panel(make_napari_viewer)
    index = _arteriole_entry(perturbations)

    perturbations.editors()[index].save.native.click()

    saved = read_saved_perturbation(tmp_path / "art_dilate_20.yaml")
    assert saved.entry == {"name": "art_dilate_20", "type": A_TYPE, "overrides": {A_PERCENT: 20}}
    assert first._haemolynx_report().startswith(
        "Saved the perturbation art_dilate_20 (arteriole constriction/dilation) in "
    )
    menu = perturbations.saved_menu
    assert menu.itemText(menu.findData("art_dilate_20")) == (
        "art_dilate_20 (arteriole constriction/dilation)"
    )
    assert menu.currentIndex() == 0

    second, later = _saved_panel(make_napari_viewer)
    later.add()
    later.choose_type(0, "pressure_sweep")
    baseline = second._haemolynx_values()[A_PERCENT]

    _choose_saved(later, "art_dilate_20")

    assert [entry["name"] for entry in later.entries()] == ["perturbation_1", "art_dilate_20"]
    assert later.entries()[1] == {"name": "art_dilate_20", "type": A_TYPE, "overrides": {A_PERCENT: 20.0}}
    editor = later.editors()[1]
    assert editor.type.value == A_TYPE
    assert editor.editors[A_PERCENT].value == 20
    values = second._haemolynx_values()
    assert values["perturbations"] == later.entries()
    assert values[A_PERCENT] == baseline
    assert second._haemolynx_report().startswith("Added the saved perturbation art_dilate_20\n")
    assert later.saved_menu.currentIndex() == 0


def test_one_added_twice_is_numbered_on(make_napari_viewer, monkeypatch, tmp_path):
    from haemolynx.pipeline.saved_perturbations import PERTURBATION_DIR_ENV

    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(tmp_path))
    widget, perturbations = _saved_panel(make_napari_viewer)
    _arteriole_entry(perturbations)
    assert perturbations.save(0) is True

    assert perturbations.add_saved("art_dilate_20") is True

    assert [entry["name"] for entry in perturbations.entries()] == ["art_dilate_20", "art_dilate_20_2"]
    assert widget._haemolynx_report().startswith(
        "Added the saved perturbation art_dilate_20 as art_dilate_20_2, as another already has its name"
    )


def test_saving_asks_before_replacing_one_saved_under_the_same_name(
    make_napari_viewer, monkeypatch, tmp_path
):
    from qtpy.QtWidgets import QMessageBox

    from haemolynx.pipeline.saved_perturbations import PERTURBATION_DIR_ENV, read_saved_perturbation

    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(tmp_path))
    _widget, perturbations = _saved_panel(make_napari_viewer)
    index = _arteriole_entry(perturbations)
    assert perturbations.save(index) is True
    perturbations.editors()[index].editors[A_PERCENT].value = -30
    asked = []
    # Only the title kept: holding the parent window would leak the viewer.
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **_k: asked.append(a[1]) or QMessageBox.No)
    )

    assert perturbations.save(index) is False

    assert asked == ["Replace the saved perturbation?"]
    path = tmp_path / "art_dilate_20.yaml"
    assert read_saved_perturbation(path).entry["overrides"] == {A_PERCENT: 20}

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_a, **_k: QMessageBox.Yes))
    assert perturbations.save(index) is True
    assert read_saved_perturbation(path).entry["overrides"] == {A_PERCENT: -30}


def test_one_with_no_type_yet_is_not_saved(make_napari_viewer, monkeypatch, tmp_path):
    from haemolynx.pipeline.saved_perturbations import PERTURBATION_DIR_ENV

    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(tmp_path))
    widget, perturbations = _saved_panel(make_napari_viewer)
    perturbations.add()

    assert perturbations.save(0) is False

    assert list(tmp_path.iterdir()) == []
    assert "no type yet" in widget._haemolynx_report()


def test_a_saved_pericyte_perturbation_leaves_this_images_mask_behind(
    make_napari_viewer, monkeypatch, tmp_path
):
    from haemolynx.gui.form import field_for
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.saved_perturbations import PERTURBATION_DIR_ENV, read_saved_perturbation

    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(tmp_path / "saved"))
    mask = tmp_path / "pericytes.tif"
    mask.write_bytes(b"")
    widget, perturbations = _saved_panel(make_napari_viewer)
    perturbations.add()
    perturbations.editors()[0].name.value = "tight"
    perturbations.choose_type(0, "pericyte_diameter_change")
    editor = perturbations.editors()[0]
    editor.editors["use_pericyte_mask_constriction"].value = True
    editor.editors["pericyte_mask_path"].value = mask
    editor.editors["pericyte_constriction_factor"].value = 0.5
    assert "pericyte_mask_path" in perturbations.entries()[0]["overrides"]

    assert perturbations.save(0) is True

    held = read_saved_perturbation(tmp_path / "saved" / "tight.yaml").entry["overrides"]
    assert "pericyte_mask_path" not in held
    assert held["use_pericyte_mask_constriction"] is True
    assert held["pericyte_constriction_factor"] == 0.5
    report = widget._haemolynx_report()
    assert "Not kept, as they belong to this image" in report
    assert field_for(default_schema()["pericyte_mask_path"]).label in report


def test_the_menu_reads_the_folder_as_it_opens(make_napari_viewer, monkeypatch, tmp_path):
    """One saved from another panel, or copied in by hand, is listed."""
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.saved_perturbations import PERTURBATION_DIR_ENV, save_perturbation

    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(tmp_path))
    _widget, perturbations = _saved_panel(make_napari_viewer)
    menu = perturbations.saved_menu
    assert menu.findData("copied_in") == -1
    save_perturbation(
        {"name": "copied_in", "type": A_TYPE, "overrides": {A_PERCENT: 5}}, default_schema()
    )

    menu.showPopup()
    menu.hidePopup()

    assert menu.findData("copied_in") > 0
    assert menu.currentIndex() == 0


def test_one_gone_from_disk_says_so_and_adds_nothing(make_napari_viewer, monkeypatch, tmp_path):
    from haemolynx.pipeline.saved_perturbations import PERTURBATION_DIR_ENV

    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(tmp_path))
    widget, perturbations = _saved_panel(make_napari_viewer)
    _arteriole_entry(perturbations, name="gone")
    perturbations.save(0)
    (tmp_path / "gone.yaml").unlink()

    _choose_saved(perturbations, "gone")

    assert len(perturbations.entries()) == 1
    assert widget._haemolynx_report().startswith("Could not add the saved perturbation")
    assert perturbations.saved_menu.findData("gone") == -1


def test_deleting_one_from_the_menu_asks_which_and_then_again(make_napari_viewer, monkeypatch, tmp_path):
    from qtpy.QtWidgets import QInputDialog, QMessageBox

    from haemolynx.gui.perturbation_editing import DELETE_SAVED
    from haemolynx.pipeline.saved_perturbations import PERTURBATION_DIR_ENV

    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(tmp_path))
    widget, perturbations = _saved_panel(make_napari_viewer)
    menu = perturbations.saved_menu
    assert menu.findData(DELETE_SAVED) == -1
    _arteriole_entry(perturbations, name="keep")
    _arteriole_entry(perturbations, name="drop")
    perturbations.save(0)
    perturbations.save(1)
    assert menu.findData(DELETE_SAVED) == menu.count() - 1

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *_a, **_k: ("drop", True)))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_a, **_k: QMessageBox.No))
    _choose_saved(perturbations, DELETE_SAVED)
    assert (tmp_path / "drop.yaml").is_file()

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_a, **_k: QMessageBox.Yes))
    _choose_saved(perturbations, DELETE_SAVED)
    assert not (tmp_path / "drop.yaml").exists()
    assert (tmp_path / "keep.yaml").is_file()
    assert menu.findData("drop") == -1
    assert menu.currentIndex() == 0
    assert widget._haemolynx_report().startswith("Deleted the saved perturbation drop")
    # The run's own list is not touched.
    assert [entry["name"] for entry in perturbations.entries()] == ["keep", "drop"]

    assert perturbations.delete_saved("keep") is True
    assert menu.findData(DELETE_SAVED) == -1


def test_the_menu_opens_the_folder_they_are_kept_in(make_napari_viewer, monkeypatch, tmp_path):
    from haemolynx.gui import _widget as widget_module
    from haemolynx.gui.perturbation_editing import OPEN_SAVED_FOLDER
    from haemolynx.pipeline.saved_perturbations import PERTURBATION_DIR_ENV

    folder = tmp_path / "not yet made"
    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(folder))
    opened = []
    monkeypatch.setattr(widget_module, "_open_in_system", lambda path: opened.append(path) or True)
    widget, perturbations = _saved_panel(make_napari_viewer)

    _choose_saved(perturbations, OPEN_SAVED_FOLDER)

    assert opened == [folder]
    assert folder.is_dir()
    assert widget._haemolynx_report() == f"Opened {folder}"
    assert perturbations.saved_menu.currentIndex() == 0


def test_the_saved_perturbation_controls_sit_on_the_tab_with_their_hover_text(panel):
    from qtpy.QtCore import Qt

    from haemolynx.gui.perturbation_editing import (
        OPEN_SAVED_FOLDER,
        OPEN_SAVED_FOLDER_TOOLTIP,
        SAVE_TOOLTIP,
        SAVED_MENU_TOOLTIP,
    )

    widget, _viewer, perturbations = panel
    perturbations.add()
    editor = perturbations.editors()[0]
    page = tab_page(widget, TAB_TITLE)

    assert page.isAncestorOf(perturbations.saved_menu)
    assert editor.container.native.isAncestorOf(editor.save.native)
    assert editor.container.native.isAncestorOf(editor.remove.native)
    assert editor.save.tooltip == SAVE_TOOLTIP
    menu = perturbations.saved_menu
    assert menu.toolTip() == SAVED_MENU_TOOLTIP
    assert menu.itemData(menu.findData(OPEN_SAVED_FOLDER), Qt.ToolTipRole) == OPEN_SAVED_FOLDER_TOOLTIP
