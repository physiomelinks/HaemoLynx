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
    settings_widget,
)
from haemolynx.gui.tabs import STAGES  # noqa: E402
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


def test_there_is_one_tab_per_stage(panel):
    from qtpy.QtWidgets import QTabWidget

    widget, _viewer = panel
    tab_widget = widget.findChild(QTabWidget)
    assert tab_widget is not None
    assert tab_widget.count() == len(STAGES)
    assert [tab_widget.tabText(i) for i in range(tab_widget.count())] == [
        stage.title for stage in STAGES
    ]


def test_a_long_tab_asks_for_far_less_room_than_its_contents_need(panel):
    """The Diameters tab has 39 rows; it must scroll, not stretch the window.

    What matters is not that every tab asks for the same height -- a scroll
    area's hint does vary a little -- but that a tab asks for much less than
    the rows inside it would need, so napari sizes the dock to the panel rather
    than to 39 spin boxes.
    """
    from qtpy.QtWidgets import QScrollArea, QTabWidget

    widget, _viewer = panel
    tab_widget = widget.findChild(QTabWidget)
    tallest_content = 0
    for index in range(tab_widget.count()):
        page = tab_widget.widget(index)
        assert isinstance(page, QScrollArea), "each tab must be scrollable"
        content_height = page.widget().sizeHint().height()
        tallest_content = max(tallest_content, content_height)
        assert page.sizeHint().height() <= content_height + 40, (
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

    `hold_ide_plots_open` only does anything while `show_plots_in_ide` is on,
    so turning it off as well made the schema warn on an untouched panel.
    """
    widget, _viewer = panel
    schema = default_schema()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema.validate(widget._haemolynx_values())

    assert [str(warning.message) for warning in caught] == []


def test_the_settings_that_open_a_browser_start_off(panel):
    """`show_plots_in_ide` makes plotly open a web browser mid-run."""
    widget, _viewer = panel
    values = widget._haemolynx_values()
    for name, expected in DISPLAY_SETTINGS_OFF_IN_NAPARI.items():
        assert values[name] == expected, f"{name} should start {expected}"


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
    assert bars.stage_bar.maximum() == len(STAGES)


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
            total=len(STAGES),
        )
    )

    assert bars.stage_bar.value() == 3
    assert bars.stage_bar.maximum() == len(STAGES)


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
            total=len(STAGES),
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
            total=len(STAGES),
        )
    )

    bars.finish()

    assert bars.stage_bar.value() == bars.stage_bar.maximum() == len(STAGES)
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
            total=len(STAGES),
        )
    )

    bars.fail("Failed: ValueError")

    assert bars.stage_bar.value() == 2
    assert bars.stage_bar.format() == "Failed: ValueError"


# --- a run, end to end -------------------------------------------------------


@pytest.mark.slow
def test_the_panel_runs_the_pipeline_on_the_open_layer(make_napari_viewer, tmp_path):
    """The whole path: layer -> settings -> eight stages -> a graph."""
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
    values["starting_node_selection_method"] = "volume"
    values["output_node_selection_method"] = "volume"
    values["starting_node_volumes"] = [INLET_BOX]
    values["output_node_volumes"] = [OUTLET_BOX]
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
    values["starting_node_selection_method"] = "volume"
    values["output_node_selection_method"] = "volume"
    values["starting_node_volumes"] = [INLET_BOX]
    values["output_node_volumes"] = [OUTLET_BOX]
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

    `resistance_pipeline_config.yaml` ships with `input_path` pointing at
    `examples/images/brain_microvessels.tiff`, which is not in the repository,
    and a config written on one machine is routinely opened on another. Neither
    is a reason to refuse to open it: the image a run works on is the layer
    already open in napari. Paths are checked when a run starts instead.

    Loading it used to be impossible two ways over -- through the "Run a saved
    config" widget, which ran preflight first and stopped at "FAILED:
    input_path: checked: examples/images/brain_microvessels.tiff", and through
    "Load config...", which assigned the file's unset values straight onto the
    widgets and raised TypeError on the first None.
    """
    make_napari_viewer()
    panel = settings_widget()

    assert not (REPO_ROOT / "examples" / "images" / "brain_microvessels.tiff").exists(), (
        "this test is only meaningful while that image is absent"
    )

    panel._haemolynx_load_config(RESISTANCE_CONFIG)

    values = panel._haemolynx_values()
    default_schema().validate(values)
    assert Path(values["input_path"]).name == "brain_microvessels.tiff"


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
                "input_p_bc": 1234.0},
    )

    panel._haemolynx_load_config(config)

    values = panel._haemolynx_values()
    assert values["min_stub_length"] == 42.5
    assert values["input_p_bc"] == 1234.0


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
