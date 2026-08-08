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
    run_config_widget,
    settings_widget,
)
from haemolynx.gui.tabs import STAGES  # noqa: E402
from haemolynx.pipeline import default_schema  # noqa: E402

pytestmark = pytest.mark.gui

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Small enough to skeletonise, build and solve in seconds.
FIXTURE = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"


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


def test_the_run_a_config_panel_builds():
    assert run_config_widget() is not None


def test_there_is_one_tab_per_stage(panel):
    from qtpy.QtWidgets import QTabWidget

    widget, _viewer = panel
    tab_widget = widget.findChild(QTabWidget)
    assert tab_widget is not None
    assert tab_widget.count() == len(STAGES)
    assert [tab_widget.tabText(i) for i in range(tab_widget.count())] == [
        stage.title for stage in STAGES
    ]


def test_a_long_tab_does_not_force_the_panel_wide_or_tall(panel):
    """The Diameters tab has 39 rows; it must scroll, not stretch the window."""
    from qtpy.QtWidgets import QScrollArea, QTabWidget

    widget, _viewer = panel
    tab_widget = widget.findChild(QTabWidget)
    heights = []
    for index in range(tab_widget.count()):
        page = tab_widget.widget(index)
        assert isinstance(page, QScrollArea), "each tab must be scrollable"
        heights.append(page.sizeHint().height())
    # A scroll area's hint does not grow with its contents, so the tab with 39
    # rows must not ask for more room than the one with three.
    assert max(heights) - min(heights) < 200, (
        f"tab size hints vary with content: {heights}"
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
    values["starting_node_selection_method"] = "all_degree_1"
    values["output_node_selection_method"] = "all_degree_1"
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
    values["starting_node_selection_method"] = "all_degree_1"
    values["output_node_selection_method"] = "all_degree_1"
    schema = default_schema()

    run_pipeline_stages(resolve_settings(values, schema=schema, config_path=None), schema)

    assert shown == [], "the run called fig.show(), which opens a browser tab"
