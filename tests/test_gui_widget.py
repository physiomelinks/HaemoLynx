"""The panel, built for real, in a real napari viewer.

Every other GUI test checks a decision without a display: which widget a setting
gets, what its options are, how its value reads back. Three bugs got past all of
them, because each only appeared when magicgui actually built the widget --
options a LineEdit will not take, an empty string a LiteralEvalLineEdit cannot
parse, a scroll area that stretched the window. These tests construct the
panel, so that class of failure is caught here rather than by opening napari.

They need napari, a Qt binding and a display, so they are marked `gui` and
skipped everywhere those are missing. CI runs them on 3.11 under xvfb.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    DISPLAY_SETTINGS_OFF_IN_NAPARI,
    OURS,
    settings_widget,
)
from haemolynx.gui.progress import TOTAL_STAGES  # noqa: E402
from haemolynx.gui.tabs import tab_titles  # noqa: E402
from haemolynx.pipeline import default_schema  # noqa: E402
from haemolynx.pipeline.progress import (  # noqa: E402
    STAGE_FINISHED,
    STAGE_STARTED,
    STEP,
    ProgressEvent,
)

pytestmark = pytest.mark.gui

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Small enough to skeletonise, build and solve in seconds. It is a grayscale
#: volume of 74 levels, which the loader thresholds at half the dtype range --
#: so it needs no preparation here. (Binarising it by hand with `> 0` would
#: give a solid block, since 95% of its voxels are nonzero.)
FIXTURE = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"


#: Boxes in physical (z, y, x) microns that split this fixture's terminals into
#: two non-empty, disjoint sets. Its vessels span y = 5..42 of a 0..47 volume,
#: so they never reach the image border and any band measured against the
#: image extent -- edge_percent's default 10% is y < 4.7 and y > 42.3 -- finds
#: nothing at all.
INLET_BOX = ((0.0, 0.0, 0.0), (47.0, 15.0, 47.0))
OUTLET_BOX = ((0.0, 35.0, 0.0), (47.0, 47.0, 47.0))


@pytest.fixture
def panel(make_napari_viewer):
    """The panel, built against a real viewer."""
    viewer = make_napari_viewer()
    return settings_widget(napari_viewer=viewer), viewer


# --- it builds at all --------------------------------------------------------


def test_the_panel_builds(panel):
    """Construction is where every widget-level mistake shows up."""
    widget, _viewer = panel
    assert widget is not None


def test_the_panel_builds_with_no_viewer():
    """Someone may call this outside napari; it must not require a viewer."""
    assert settings_widget(napari_viewer=None) is not None


def test_there_is_one_tab_per_stage_that_opens_one(panel):
    """Not one per stage: `solve` shows its rows on the haemodynamics tab."""
    from qtpy.QtWidgets import QTabWidget

    widget, _viewer = panel
    tab_widget = widget.findChild(QTabWidget)
    assert tab_widget is not None
    assert [tab_widget.tabText(i) for i in range(tab_widget.count())] == list(
        tab_titles()
    )


def test_a_long_tab_asks_for_far_less_room_than_its_contents_need(panel):
    """The Diameters tab has 39 rows; it must scroll, not stretch the window.

    What matters is not that every tab asks for the same height -- a scroll
    area's hint does vary a little -- but that a tab asks for much less than
    the rows inside it would need, so napari sizes the dock to the panel rather
    than to 39 spin boxes.
    """
    from qtpy.QtWidgets import QScrollArea, QTabWidget

    from haemolynx.gui._widget import TAB_SCROLL_HINT_HEIGHT

    widget, _viewer = panel
    tab_widget = widget.findChild(QTabWidget)
    tallest_content = 0
    for index in range(tab_widget.count()):
        page = tab_widget.widget(index)
        assert isinstance(page, QScrollArea), "each tab must be scrollable"
        content_height = page.widget().sizeHint().height()
        tallest_content = max(tallest_content, content_height)
        assert page.sizeHint().height() <= TAB_SCROLL_HINT_HEIGHT + 8, (
            f"tab {tab_widget.tabText(index)} asks for "
            f"{page.sizeHint().height()}px for {content_height}px of content"
        )

    # The longest tab's contents are far taller than any tab asks for.
    asked = max(
        tab_widget.widget(index).sizeHint().height()
        for index in range(tab_widget.count())
    )
    assert tallest_content > 800, "the fixture no longer has a long tab to test"
    assert asked < tallest_content / 2, (
        f"the tallest tab asks for {asked}px against {tallest_content}px of content"
    )


def test_the_panel_does_not_demand_more_height_than_a_1080p_work_area(panel):
    """Windows cannot honour a min height taller than the usable screen.

    A 1920x1200 Dell with a taskbar leaves ~1009px of client height. Qt then
    warns ``QWindowsWindow::setGeometry`` if the main window minimum is 1057
    because the dock's size hint was the full tab contents. Tabs must report a
    bounded hint and the panel's own minimum must stay below that work area.
    """
    from qtpy.QtWidgets import QScrollArea, QTabWidget

    from haemolynx.gui._widget import TAB_SCROLL_HINT_HEIGHT

    widget, _viewer = panel
    tab_widget = widget.findChild(QTabWidget)
    for index in range(tab_widget.count()):
        page = tab_widget.widget(index)
        assert isinstance(page, QScrollArea)
        assert page.sizeHint().height() <= TAB_SCROLL_HINT_HEIGHT + 8, (
            f"tab {tab_widget.tabText(index)} asks for {page.sizeHint().height()}px"
        )

    # Menu, title bar and taskbar take the rest of a 1080p screen.
    assert widget.minimumSizeHint().height() < 900, (
        f"panel minimum height {widget.minimumSizeHint().height()}px "
        "will not fit a 1080p work area"
    )
    assert widget.sizeHint().height() < 900, (
        f"panel size hint {widget.sizeHint().height()}px "
        "will not fit a 1080p work area"
    )


# --- what it starts with -----------------------------------------------------


def test_an_untouched_panel_reads_back_the_schema_defaults(panel):
    """Reading the widgets must give the settings, not the widgets' fallbacks."""
    widget, _viewer = panel
    schema = default_schema()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = widget._haemolynx_values()
        resolved = schema.validate(values)

    assert [str(w.message) for w in caught] == []
    expected = schema.validate(
        {
            setting.name: DISPLAY_SETTINGS_OFF_IN_NAPARI.get(
                setting.name, setting.default
            )
            for setting in schema
        }
    )
    assert resolved == expected


def test_the_panel_does_not_set_a_setting_nothing_will_read(panel):
    """Switching a display setting off must not earn an ineffective warning.

    `hold_ide_plots_open` only does anything while Produce IDE plots and
    `show_plots_in_ide` are on, so turning those nested rows off as well
    made the schema warn on an untouched panel.
    """
    widget, _viewer = panel
    schema = default_schema()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema.validate(widget._haemolynx_values())

    assert [str(warning.message) for warning in caught] == []


def test_the_settings_that_open_a_browser_start_off(panel):
    """Produce IDE plots starts off so napari does not write Plotly HTML."""
    widget, _viewer = panel
    values = widget._haemolynx_values()
    for name, expected in DISPLAY_SETTINGS_OFF_IN_NAPARI.items():
        assert values[name] == expected, f"{name} should start {expected}"
    assert values["show_plots_in_ide"] is False
    assert values["interactive_plots"] is False


def test_the_input_path_starts_empty(panel):
    widget, _viewer = panel
    assert widget._haemolynx_values()["input_path"] is None


# --- the layer already open --------------------------------------------------


def test_a_layer_open_before_the_panel_becomes_the_input(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.open(str(FIXTURE))

    widget = settings_widget(napari_viewer=viewer)

    assert Path(widget._haemolynx_values()["input_path"]) == FIXTURE


def test_a_layer_added_after_the_panel_becomes_the_input(panel):
    widget, viewer = panel
    assert widget._haemolynx_values()["input_path"] is None

    viewer.open(str(FIXTURE))

    assert Path(widget._haemolynx_values()["input_path"]) == FIXTURE


def test_a_layer_with_a_scale_sets_the_voxel_size(panel):
    """napari scales (z, y, x); the setting is (x, y, z)."""
    widget, viewer = panel
    (layer,) = viewer.open(str(FIXTURE))
    layer.scale = (2.0, 0.5, 0.4)
    # Re-select to re-apply, as choosing it in the panel would.
    viewer.layers.selection.active = layer
    rebuilt = settings_widget(napari_viewer=viewer)

    values = rebuilt._haemolynx_values()
    assert values["voxel_size_override_xyz"] == [0.4, 0.5, 2.0]
    assert values["voxel_size_policy"] == "override"


# --- the progress bars -------------------------------------------------------
#
# What the bars should read is decided by `haemolynx.gui.progress`, which
# `test_gui_progress.py` covers without a display. What is left for here is the
# wiring: that the panel has bars at all, and that an event moves them.


def test_the_panel_has_a_bar_for_the_stages_and_one_for_the_steps(panel):
    from qtpy.QtWidgets import QProgressBar

    widget, _viewer = panel
    assert len(widget.findChildren(QProgressBar)) == 2


def test_the_bars_are_hidden_until_a_run_starts(panel):
    widget, _viewer = panel
    bars = widget._haemolynx_progress
    assert bars.stage_bar.isHidden()
    assert bars.step_bar.isHidden()


def test_starting_a_run_shows_an_empty_stage_bar(panel):
    widget, _viewer = panel
    bars = widget._haemolynx_progress

    bars.start()

    assert not bars.stage_bar.isHidden()
    assert bars.stage_bar.value() == 0
    assert bars.stage_bar.maximum() == TOTAL_STAGES


def test_a_finished_stage_moves_the_stage_bar(panel):
    widget, _viewer = panel
    bars = widget._haemolynx_progress
    bars.start()

    bars.show_event(
        ProgressEvent(
            kind=STAGE_FINISHED,
            stage="build_network",
            title="3. Graph",
            index=2,
            total=TOTAL_STAGES,
        )
    )

    assert bars.stage_bar.value() == 3
    assert bars.stage_bar.maximum() == TOTAL_STAGES


def test_a_graph_build_step_moves_the_second_bar(panel):
    widget, _viewer = panel
    bars = widget._haemolynx_progress
    bars.start()
    assert bars.step_bar.isHidden()

    bars.show_event(
        ProgressEvent(
            kind=STEP,
            stage="build_network",
            title="3. Graph",
            index=2,
            total=TOTAL_STAGES,
            step="collapse_node_clusters",
            step_index=4,
            step_total=11,
        )
    )

    assert not bars.step_bar.isHidden()
    assert bars.step_bar.value() == 5
    assert bars.step_bar.maximum() == 11


def test_a_finished_run_fills_the_stage_bar_and_drops_the_step_bar(panel):
    widget, _viewer = panel
    bars = widget._haemolynx_progress
    bars.start()
    bars.show_event(
        ProgressEvent(
            kind=STAGE_STARTED,
            stage="build_network",
            title="3. Graph",
            index=2,
            total=TOTAL_STAGES,
        )
    )

    bars.finish()

    assert bars.stage_bar.value() == bars.stage_bar.maximum() == TOTAL_STAGES
    assert bars.step_bar.isHidden()


def test_a_failed_run_leaves_the_bar_where_it_stopped(panel):
    """A full bar would say the run finished; it did not."""
    widget, _viewer = panel
    bars = widget._haemolynx_progress
    bars.start()
    bars.show_event(
        ProgressEvent(
            kind=STAGE_FINISHED,
            stage="skeletonise",
            title="2. Skeletonise",
            index=1,
            total=TOTAL_STAGES,
        )
    )

    bars.fail("Failed: ValueError")

    assert bars.stage_bar.value() == 2
    assert bars.stage_bar.format() == "Failed: ValueError"


# --- a run, end to end -------------------------------------------------------


@pytest.mark.slow
def test_the_panel_runs_the_pipeline_on_the_open_layer(make_napari_viewer, tmp_path):
    """The whole path: layer -> settings -> every stage -> a graph."""
    from haemolynx.pipeline import resolve_settings, run_pipeline_stages

    viewer = make_napari_viewer()
    viewer.open(str(FIXTURE))
    widget = settings_widget(napari_viewer=viewer)

    values = widget._haemolynx_values()
    values["vtk_output_prefix"] = tmp_path / "run"
    values["plot_dir"] = tmp_path / "plots"
    values["statistics"] = False
    # Inlets from one end of the volume and outlets from the other. Both set to
    # "all_degree_1" would leave outlets empty, because the outlet call excludes
    # the inlets it was already given.
    values["inlet_node_selection_method"] = "volume"
    values["outlet_node_selection_method"] = "volume"
    values["inlet_node_volumes"] = [INLET_BOX]
    values["outlet_node_volumes"] = [OUTLET_BOX]
    schema = default_schema()
    settings = resolve_settings(values, schema=schema, config_path=None)

    graph = run_pipeline_stages(settings, schema)

    assert graph is not None
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert list(tmp_path.glob("*.vtp")), "the run wrote no VTK"


@pytest.mark.slow
def test_a_run_from_the_panel_opens_no_browser(make_napari_viewer, tmp_path, monkeypatch):
    """plotly's fig.show() opens a web browser; a run from napari must not."""
    import plotly.graph_objects as go

    from haemolynx.pipeline import resolve_settings, run_pipeline_stages

    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: shown.append(self))

    viewer = make_napari_viewer()
    viewer.open(str(FIXTURE))
    widget = settings_widget(napari_viewer=viewer)

    values = widget._haemolynx_values()
    values["vtk_output_prefix"] = tmp_path / "run"
    values["plot_dir"] = tmp_path / "plots"
    values["statistics"] = False
    values["inlet_node_selection_method"] = "volume"
    values["outlet_node_selection_method"] = "volume"
    values["inlet_node_volumes"] = [INLET_BOX]
    values["outlet_node_volumes"] = [OUTLET_BOX]
    schema = default_schema()

    run_pipeline_stages(resolve_settings(values, schema=schema, config_path=None), schema)

    assert shown == [], "the run called fig.show(), which opens a browser tab"


def test_the_haemodynamics_tab_is_named_for_what_it_does(panel):
    """#125: the tab was called "Resistances".

    Named literally rather than derived from `STAGES`, unlike the test above:
    comparing the tabs against the same list they are built from cannot catch a
    title that is wrong in both places, which is exactly what a rename is.

    The stage computes Poiseuille resistances *and* applies pericyte
    constriction, and it is where `run_haemodynamics` lives -- "Resistances"
    named one output of it.
    """
    from qtpy.QtWidgets import QTabWidget

    widget, _viewer = panel
    tabs = widget.findChild(QTabWidget)
    titles = [tabs.tabText(i) for i in range(tabs.count())]

    assert "6. Haemodynamics" in titles, titles
    assert not any("Resistances" in title for title in titles), titles


# --- opening a config in the panel -------------------------------------------


#: The config the panel's own schema describes. The brain, carotid and simple
#: examples each add settings of their own, so `default_schema()` rejects them
#: by design -- the panel is the pipeline's schema, not theirs.
RESISTANCE_CONFIG = REPO_ROOT / "examples" / "resistance_pipeline_config.yaml"


def test_loading_the_resistance_config_does_not_need_its_input_image(
    make_napari_viewer,
):
    """Opening a config reads the file and nothing the file names.

    A config written on one machine is routinely opened on another, and
    `resistance_pipeline_config.yaml` shipped for a long time naming an image
    that was not in the repository at all. Neither is a reason to refuse to
    open it: the image a run works on is the layer already open in napari, and
    paths are checked when a run starts instead. `test_loading_a_config_
    naming_a_missing_image_still_fails_the_run_checks` covers the missing-file
    case directly, so this one does not depend on the shipped config staying
    broken.

    Loading it used to be impossible two ways over -- through the "Run a saved
    config" widget, which ran preflight first and stopped at "FAILED:
    input_path: checked: examples/images/brain_microvessels.tiff", and through
    "Load config...", which assigned the file's unset values straight onto the
    widgets and raised TypeError on the first None.
    """
    from haemolynx.parsers import load_config

    make_napari_viewer()
    panel = settings_widget()

    panel._haemolynx_load_config(RESISTANCE_CONFIG)

    values = panel._haemolynx_values()
    default_schema().validate(values)
    # Whatever the config names, not a filename pinned here: which image it
    # points at is that config's business and has changed once already.
    on_file = load_config(RESISTANCE_CONFIG, default_schema())["input_path"]
    assert Path(values["input_path"]) == Path(on_file)


def test_loading_a_config_whose_unset_values_are_null(make_napari_viewer, tmp_path):
    """A None in the file must reach the widget as the blank the form uses.

    Rows are built through `form.display_value_for`, which turns an unset value
    into the empty box a FileEdit will accept. Writing to a row afterwards has
    to go the same way; assigning the raw None raised
    "TypeError: value must be a string, or list/tuple of strings".
    """
    from haemolynx.parsers import dump_config

    make_napari_viewer()
    panel = settings_widget()
    schema = default_schema()

    config = tmp_path / "nulls.yaml"
    dump_config(config, schema, values={s.name: s.default for s in schema})

    panel._haemolynx_load_config(config)

    assert panel._haemolynx_values()["input_path"] is None


def test_loading_a_config_reports_it_and_applies_its_values(make_napari_viewer, tmp_path):
    """The values in the file end up in the form, not just parsed and dropped."""
    from haemolynx.parsers import dump_config

    make_napari_viewer()
    panel = settings_widget()
    schema = default_schema()

    config = tmp_path / "distinctive.yaml"
    dump_config(
        config,
        schema,
        values={**{s.name: s.default for s in schema}, "min_stub_length": 42.5,
                "inlet_p_bc": 1234.0},
    )

    panel._haemolynx_load_config(config)

    values = panel._haemolynx_values()
    assert values["min_stub_length"] == 42.5
    assert values["inlet_p_bc"] == 1234.0


def test_loading_a_config_naming_a_missing_image_still_fails_the_run_checks(
    make_napari_viewer, tmp_path
):
    """Loading is lenient; starting a run is not. Both halves matter."""
    from haemolynx.parsers import dump_config
    from haemolynx.pipeline import preflight, resolve_settings

    make_napari_viewer()
    panel = settings_widget()
    schema = default_schema()

    config = tmp_path / "missing_image.yaml"
    dump_config(
        config,
        schema,
        values={**{s.name: s.default for s in schema},
                "input_path": tmp_path / "not_here.tif"},
    )

    panel._haemolynx_load_config(config)  # loads without complaint

    settings = resolve_settings(panel._haemolynx_values(), schema=schema, config_path=None)
    result = preflight(settings, schema)
    assert not result.ok
    assert any("input_path" in message for message in result.errors)


def test_loading_a_config_with_a_long_spaced_path_round_trips(
    make_napari_viewer, tmp_path
):
    """Save then Load must keep an absolutised Windows-style path intact.

    This is the failure mode behind the napari "config won't load" report on
    Windows: FileEdit stores `C:/Users/.../My Dataset/mask.tif`, dump folded
    it at the space, and the next Load hit a YAML scanner error.
    """
    from haemolynx.parsers import dump_config

    make_napari_viewer()
    panel = settings_widget()
    schema = default_schema()

    absolute = (
        tmp_path
        / "Dropbox"
        / (
            "Composite_06082026_E14p5_clnd5_25x_940nm_1040_texasred3kkda_"
            "zstack_spot1_1p5z_MCAregion_Simple Segmentation_arteriole.tiff"
        )
    )
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(b"")

    saved = tmp_path / "panel_roundtrip.yaml"
    dump_config(
        saved,
        schema,
        values={**{s.name: s.default for s in schema}, "input_path": absolute},
    )
    assert "\n  Segmentation_arteriole.tiff" not in saved.read_text(encoding="utf-8")

    panel._haemolynx_load_config(saved)

    assert "Could not load" not in panel._haemolynx_report()
    assert Path(panel._haemolynx_values()["input_path"]) == absolute


def test_panel_save_config_round_trips_long_paths_with_spaces(
    make_napari_viewer, tmp_path
):
    """The panel's Save path must write YAML the same Load can read back.

    Exercises ``_haemolynx_save_config`` (what "Save config..." calls after the
    file dialogue) with FileEdit-absolutised Windows paths containing spaces.
    """
    make_napari_viewer()
    panel = settings_widget()
    rows = panel._haemolynx_rows()

    absolute = (
        tmp_path
        / "Dropbox"
        / (
            "Composite_06082026_E14p5_clnd5_25x_940nm_1040_texasred3kkda_"
            "zstack_spot1_1p5z_MCAregion_Simple Segmentation_.tiff"
        )
    )
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(b"")
    arteriole = absolute.with_name(
        absolute.name.replace("_.tiff", "_arteriole.tiff")
    )
    arteriole.write_bytes(b"")

    rows["input_path"].value = absolute
    rows["automated_vessel_assignment"].value = True
    rows["use_large_vessel_masks"].value = True
    rows["large_arteriole_mask_path"].value = arteriole

    saved = tmp_path / "from_panel.yaml"
    assert panel._haemolynx_save_config(saved) is True
    assert "Could not save" not in panel._haemolynx_report()
    text = saved.read_text(encoding="utf-8")
    assert "\n  Segmentation_.tiff" not in text
    assert "\n  Segmentation_arteriole.tiff" not in text

    panel._haemolynx_load_config(saved)
    assert "Could not load" not in panel._haemolynx_report()
    assert Path(panel._haemolynx_values()["input_path"]) == absolute
    assert Path(panel._haemolynx_values()["large_arteriole_mask_path"]) == arteriole


def test_panel_save_config_adds_a_yaml_suffix(make_napari_viewer, tmp_path):
    make_napari_viewer()
    panel = settings_widget()
    dest = tmp_path / "my_settings"
    assert panel._haemolynx_save_config(dest) is True
    written = tmp_path / "my_settings.yaml"
    assert written.is_file()
    assert "Wrote" in panel._haemolynx_report()
    assert written.name in panel._haemolynx_report()


def test_loading_a_config_for_a_different_schema_reports_instead_of_raising(
    make_napari_viewer,
):
    """simple_network_config.yaml is not a pipeline config; say so in the panel."""
    make_napari_viewer()
    panel = settings_widget()

    panel._haemolynx_load_config(REPO_ROOT / "examples" / "simple_network_config.yaml")

    report = panel._haemolynx_report()
    assert report.startswith("Could not load"), report
    assert "output_dir" in report


# --- the About widget --------------------------------------------------------


def test_the_about_widget_builds(make_napari_viewer):
    from haemolynx.gui._widget import about_widget

    make_napari_viewer()
    assert about_widget() is not None


def test_loading_a_config_keeps_its_paths_as_it_wrote_them(make_napari_viewer):
    """A FileEdit makes every path absolute, which rewrites the config.

    `resistance_pipeline_config.yaml` names its files relatively. Put those in
    the form and magicgui stores `/home/you/wherever/classifiers/....ilp`, so
    "Save config..." would write this machine's layout back into a portable
    file, and the values then differ from their defaults -- which is what made
    a run report fourteen settings as set while nothing reads them.
    """
    make_napari_viewer()
    panel = settings_widget()

    panel._haemolynx_load_config(RESISTANCE_CONFIG)
    values = panel._haemolynx_values()

    for name in ("input_path", "ilastik_classifier_path",
                 "ilastik_small_arteriole_classifier_path"):
        assert not Path(values[name]).is_absolute(), (
            f"{name} came back as {values[name]}, absolute, not as the config wrote it"
        )


def test_loading_a_config_does_not_make_a_run_warn_about_its_own_defaults(
    make_napari_viewer,
):
    """The symptom the path rewriting caused, pinned at the point it appeared.

    The config leaves ilastik off and still fills its paths in, which is what
    the schema calls ordinary practice. Absolutised, each of those stopped
    matching its default and a run warned about all fourteen.
    """
    from haemolynx.parsers import IneffectiveSettingWarning
    from haemolynx.pipeline import resolve_settings

    make_napari_viewer()
    panel = settings_widget()
    panel._haemolynx_load_config(RESISTANCE_CONFIG)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_settings(
            panel._haemolynx_values(), schema=default_schema(), config_path=None
        )

    ineffective = [
        w for w in caught if issubclass(w.category, IneffectiveSettingWarning)
    ]
    assert not ineffective, (
        f"{len(ineffective)} settings reported as set-but-unread: "
        f"{[str(w.message)[:60] for w in ineffective]}"
    )


def test_choosing_a_different_file_replaces_what_the_config_said(
    make_napari_viewer, tmp_path
):
    """Only the loaded value is preserved; an edit is the user's own choice."""
    make_napari_viewer()
    panel = settings_widget()
    panel._haemolynx_load_config(RESISTANCE_CONFIG)

    chosen = tmp_path / "my_own.tif"
    chosen.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = chosen

    assert Path(panel._haemolynx_values()["input_path"]) == chosen


def test_a_config_does_not_steal_the_input_from_the_open_layer(
    make_napari_viewer,
):
    """The image on screen is the input; a config names a different one.

    Every config names the image it was written for, and the shipped one names
    an image that is not in the repository. Opening it over an adopted layer
    pointed the run at that file, so "Run checks" failed on an input the user
    could see was already loaded.
    """
    viewer = make_napari_viewer()
    viewer.open(str(FIXTURE))
    panel = settings_widget(napari_viewer=viewer)

    from_layer = panel._haemolynx_values()["input_path"]
    assert Path(from_layer) == FIXTURE

    panel._haemolynx_load_config(RESISTANCE_CONFIG)

    assert Path(panel._haemolynx_values()["input_path"]) == FIXTURE
    assert "keeping" in panel._haemolynx_report(), panel._haemolynx_report()


def test_the_run_checks_pass_on_a_config_loaded_over_an_open_layer(
    make_napari_viewer, tmp_path
):
    """The whole point: this combination used to fail preflight."""
    from haemolynx.pipeline import preflight, resolve_settings

    viewer = make_napari_viewer()
    viewer.open(str(FIXTURE))
    panel = settings_widget(napari_viewer=viewer)
    panel._haemolynx_load_config(RESISTANCE_CONFIG)

    rows = panel._haemolynx_rows()
    rows["vtk_output_prefix"].value = tmp_path / "run"
    rows["base_plot_dir"].value = tmp_path / "plots"

    settings = resolve_settings(
        panel._haemolynx_values(), schema=default_schema(), config_path=None
    )
    result = preflight(settings, default_schema())
    assert result.ok, result.errors


def test_a_config_still_supplies_the_input_when_no_layer_is_open(
    make_napari_viewer,
):
    """With nothing adopted there is nothing to defer to, so the file wins."""
    make_napari_viewer()
    panel = settings_widget()

    panel._haemolynx_load_config(RESISTANCE_CONFIG)

    from haemolynx.parsers import load_config

    on_file = load_config(RESISTANCE_CONFIG, default_schema())["input_path"]
    assert Path(panel._haemolynx_values()["input_path"]) == Path(on_file)
    assert "keeping" not in panel._haemolynx_report()


def test_a_config_still_applies_everything_other_than_the_input(
    make_napari_viewer,
):
    """Only the input is held back; the rest of the file is the point of it."""
    viewer = make_napari_viewer()
    viewer.open(str(FIXTURE))
    panel = settings_widget(napari_viewer=viewer)

    from haemolynx.parsers import load_config

    on_file = load_config(RESISTANCE_CONFIG, default_schema())
    panel._haemolynx_load_config(RESISTANCE_CONFIG)
    values = panel._haemolynx_values()

    for name in ("min_stub_length", "inlet_p_bc", "skeleton_bridge_gap_size"):
        assert values[name] == on_file[name], name


def test_showing_our_own_mask_layer_does_not_hijack_the_input(make_napari_viewer):
    """A run's own results layers must never become the next run's input.

    `vessel_mask_volume_layers` shows the large-vessel masks as Image-kind
    layers (results.py). Adding any Image layer used to be treated as "the
    user just dropped this in, use it as input" -- so showing those masks
    silently repointed input_path at the mask's own exported file. Without a
    full napari restart to wipe that state, every later run then used the
    mask instead of the real segmentation. Only a layer the user actually
    brings in should ever become the input.
    """
    import numpy as np

    viewer = make_napari_viewer()
    viewer.open(str(FIXTURE))
    panel = settings_widget(napari_viewer=viewer)
    assert Path(panel._haemolynx_values()["input_path"]) == FIXTURE

    viewer.add_image(
        np.zeros((4, 4, 4), dtype=np.float32),
        name="HaemoLynx - large venule mask",
        metadata={OURS: {"kind": "image"}},
    )

    assert Path(panel._haemolynx_values()["input_path"]) == FIXTURE


# --- an adopted layer is put in the same frame as the results ----------------


ANISOTROPIC_XYZ = (0.4, 0.5, 2.0)
ANISOTROPIC_ZYX = (2.0, 0.5, 0.4)


def _aniso_tiff(path):
    import numpy as np
    import tifffile

    tifffile.imwrite(
        path,
        np.zeros((6, 12, 16), dtype=np.uint8),
        imagej=True,
        resolution=(1.0 / ANISOTROPIC_XYZ[0], 1.0 / ANISOTROPIC_XYZ[1]),
        metadata={"spacing": ANISOTROPIC_XYZ[2], "unit": "um"},
    )
    return path


def test_adopting_a_layer_scales_it_from_its_own_file(make_napari_viewer, tmp_path):
    """napari's readers ignore a TIFF's resolution tags, so a dragged-in stack
    sits at one unit per voxel while everything the run draws is in microns.

    On the nerve stack that is 2.029 um of z drawn as 1: the image ends up at
    58% of its depth and the vessels do not lie on the vessels.
    """
    viewer = make_napari_viewer()
    viewer.open(str(_aniso_tiff(tmp_path / "aniso.tif")))
    layer = viewer.layers[0]
    assert tuple(float(s) for s in layer.scale) == (1.0, 1.0, 1.0)

    settings_widget(napari_viewer=viewer)

    assert tuple(float(s) for s in layer.scale) == pytest.approx(ANISOTROPIC_ZYX)


def test_the_scale_it_applies_is_the_one_the_run_will_use(make_napari_viewer, tmp_path):
    """The whole point is that the two end up in one frame.

    The run scales its own image layer by `voxel_size_zyx`; if the adopted
    layer were scaled by anything else the two copies would still not overlay.
    """
    from haemolynx.io import load_3d_tif_with_voxel_size, voxel_size_zyx_from_xyz

    path = _aniso_tiff(tmp_path / "aniso.tif")
    viewer = make_napari_viewer()
    viewer.open(str(path))
    settings_widget(napari_viewer=viewer)

    _image, x, y, z, _meta = load_3d_tif_with_voxel_size(str(path))
    assert tuple(float(s) for s in viewer.layers[0].scale) == pytest.approx(
        voxel_size_zyx_from_xyz((x, y, z))
    )


def test_a_layer_that_already_has_a_scale_is_left_alone(make_napari_viewer, tmp_path):
    """Someone who set a scale meant it, and it already reaches the settings."""
    viewer = make_napari_viewer()
    viewer.open(str(_aniso_tiff(tmp_path / "aniso.tif")))
    layer = viewer.layers[0]
    layer.scale = (3.0, 3.0, 3.0)

    panel = settings_widget(napari_viewer=viewer)

    assert tuple(float(s) for s in layer.scale) == (3.0, 3.0, 3.0)
    # And it is what the run is told to use, as before.
    assert panel._haemolynx_values()["voxel_size_override_xyz"] == [3.0, 3.0, 3.0]


def test_a_file_with_no_voxel_metadata_leaves_the_layer_alone(
    make_napari_viewer, tmp_path
):
    import numpy as np
    import tifffile

    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((6, 12, 16), dtype=np.uint8))
    viewer = make_napari_viewer()
    viewer.open(str(path))

    settings_widget(napari_viewer=viewer)

    assert tuple(float(s) for s in viewer.layers[0].scale) == (1.0, 1.0, 1.0)


def test_scaling_the_layer_does_not_change_what_the_run_is_told(
    make_napari_viewer, tmp_path
):
    """Display only. The run reads the same file and finds the same metadata,
    so pinning an override here would be a second source for one number."""
    viewer = make_napari_viewer()
    viewer.open(str(_aniso_tiff(tmp_path / "aniso.tif")))

    panel = settings_widget(napari_viewer=viewer)

    values = panel._haemolynx_values()
    assert values["voxel_size_override_xyz"] is None
    assert values["voxel_size_policy"] != "override"


def test_shared_ilastik_reparent_does_not_spawn_floating_windows(panel):
    """Shared ilastik rows move between Input and Boundaries.

    Showing a magicgui row with no Qt parent opens a top-level window next to
    napari. Toggling the host must never leave those rows visible while
    unparented — the same class of bug as Boundaries ``place_shared``.

    While the host tab is not current, magicgui may keep ``visible`` False
    even though the row is correctly parented; switch to that tab before
    asserting the rows appear.
    """
    from qtpy.QtWidgets import QApplication

    from haemolynx.gui.form import SHARED_ILASTIK_SETTINGS

    widget, _viewer = panel
    widget.show()
    rows = widget._haemolynx_rows()
    tabs = widget._haemolynx_tabs
    before = {id(w) for w in QApplication.topLevelWidgets() if w.isVisible()}

    def assert_no_new_windows() -> None:
        QApplication.processEvents()
        appeared = [
            w
            for w in QApplication.topLevelWidgets()
            if w.isVisible() and id(w) not in before
        ]
        assert appeared == []

    def assert_hosted_and_embedded() -> None:
        for name in SHARED_ILASTIK_SETTINGS:
            native = rows[name].native
            assert native.parent() is not None, name
            assert not native.isWindow(), name

    def select_tab(title_substring: str) -> None:
        for index in range(tabs.count()):
            if title_substring in tabs.tabText(index):
                tabs.setCurrentIndex(index)
                QApplication.processEvents()
                return
        raise AssertionError(f"no tab containing {title_substring!r}")

    rows["use_ilastik_segmentation"].value = True
    QApplication.processEvents()
    select_tab("Input")
    assert_hosted_and_embedded()
    for name in SHARED_ILASTIK_SETTINGS:
        assert rows[name].visible is True, name
    assert_no_new_windows()

    rows["use_ilastik_segmentation"].value = False
    QApplication.processEvents()
    for name in SHARED_ILASTIK_SETTINGS:
        assert rows[name].visible is False, name
    assert_no_new_windows()

    rows["automated_vessel_assignment"].value = True
    rows["use_large_vessel_masks"].value = True
    rows["use_ilastik_large_vessel_segmentation"].value = True
    QApplication.processEvents()
    assert_hosted_and_embedded()
    assert_no_new_windows()
    select_tab("Boundaries")
    assert_hosted_and_embedded()
    for name in SHARED_ILASTIK_SETTINGS:
        assert rows[name].visible is True, name
    assert_no_new_windows()

    rows["use_ilastik_segmentation"].value = True
    QApplication.processEvents()
    select_tab("Input")
    assert_hosted_and_embedded()
    for name in SHARED_ILASTIK_SETTINGS:
        assert rows[name].visible is True, name
    assert_no_new_windows()

    rows["use_ilastik_segmentation"].value = False
    rows["use_ilastik_large_vessel_segmentation"].value = False
    QApplication.processEvents()
    for name in SHARED_ILASTIK_SETTINGS:
        assert rows[name].visible is False, name
    assert_no_new_windows()


def test_thick_vessel_threshold_overrides_show_auto_placeholder_and_nest(panel):
    """The two threshold overrides hint 'auto' when empty and hide with their parent.

    Regression for a GUI request: an empty box for these two used to give no
    indication of what would actually be used. While the Skeletonise tab is
    not current, magicgui may keep ``visible`` False regardless of prerequisite
    state, so the tab must be selected before the nesting assertions mean
    anything (see ``test_shared_ilastik_knobs_...`` above).
    """
    from qtpy.QtWidgets import QApplication, QTabWidget

    widget, _viewer = panel
    widget.show()
    rows = widget._haemolynx_rows()
    tabs = widget.findChild(QTabWidget)
    for index in range(tabs.count()):
        if "Skeletonise" in tabs.tabText(index):
            tabs.setCurrentIndex(index)
            QApplication.processEvents()
            break
    else:
        raise AssertionError("no tab containing 'Skeletonise'")

    wall = rows["skeleton_thick_vessel_wall_absorption_um"]
    flake = rows["skeleton_thick_vessel_flake_filter_um"]

    assert wall.value == "" and flake.value == ""
    assert wall.native.placeholderText() == "auto"
    assert flake.native.placeholderText() == "auto"

    rows["use_thick_vessel_skeletonisation"].value = False
    QApplication.processEvents()
    assert wall.visible is False
    assert flake.visible is False
    assert rows["skeleton_thick_vessel_min_radius_um"].visible is False
    assert rows["skeleton_fill_mask_holes_before_thickness"].visible is False

    rows["use_thick_vessel_skeletonisation"].value = True
    QApplication.processEvents()
    assert wall.visible is True
    assert flake.visible is True
    assert rows["skeleton_thick_vessel_min_radius_um"].visible is True
    assert rows["skeleton_fill_mask_holes_before_thickness"].visible is True


def test_thick_vessel_row_relabels_when_large_vessel_network_mode_is_on(panel):
    """Once thick-vessel skeletonisation and the large-vessel-network mode
    are both on, the checkbox's own action is no longer "thick vessel
    skeletonisation" -- it becomes "also segment large vessels". Turning
    either back off must revert the label."""
    from qtpy.QtWidgets import QApplication

    widget, _viewer = panel
    widget.show()
    rows = widget._haemolynx_rows()
    original_label = rows["use_thick_vessel_skeletonisation"].label

    rows["use_thick_vessel_skeletonisation"].value = True
    rows["use_large_vessel_masks"].value = True
    rows["automated_vessel_assignment"].value = True
    rows["cut_network_at_large_vessel_volumes"].value = False
    QApplication.processEvents()
    assert rows["use_thick_vessel_skeletonisation"].label == original_label

    rows["assign_large_vessel_branch_orders"].value = True
    QApplication.processEvents()
    assert rows["use_thick_vessel_skeletonisation"].label == "Also segment large vessels"

    rows["assign_large_vessel_branch_orders"].value = False
    QApplication.processEvents()
    assert rows["use_thick_vessel_skeletonisation"].label == original_label


