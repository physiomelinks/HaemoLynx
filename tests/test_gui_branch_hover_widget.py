"""Branch-hover layer registration and metrics panel in a real napari viewer."""
from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402
from haemolynx.gui._widget import (  # noqa: E402
    _apply_layers,
    _branch_hover_mouse_move,
    _layer_controls,
)
from haemolynx.gui.branch_hover import format_metric_value  # noqa: E402
from haemolynx.gui.results import BRANCH_HOVER, ResultLayers  # noqa: E402
from test_gui_results import a_graph, network  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _reset_session_selection():
    widget_mod._branch_hover_session_selected = None
    yield
    widget_mod._branch_hover_session_selected = None


def _draw_hover(viewer, graph=None):
    group = ResultLayers().stage_finished(
        "build_network", network(graph or a_graph())
    )
    _apply_layers(viewer, group)
    return viewer.layers[BRANCH_HOVER]


def test_branch_hover_layer_registers_with_tooltip_features(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = _draw_hover(viewer)
    assert isinstance(layer, napari.layers.Points)
    assert "tooltip" in layer.features
    assert "branch_id" in layer.features
    assert str(layer.features["tooltip"][0]).startswith("branchID: ")
    assert _branch_hover_mouse_move in layer.mouse_move_callbacks


def test_branch_hover_panel_offers_only_available_metrics(make_napari_viewer):
    from haemolynx.gui._widget import settings_widget

    viewer = make_napari_viewer()
    settings_widget(napari_viewer=viewer)
    layer = _draw_hover(viewer, a_graph())
    controls = _layer_controls(viewer, layer)
    assert controls is not None
    panel = controls._haemolynx_branch_hover
    assert panel.offered == ("tortuosity", "length")
    assert set(panel._boxes) == {"tortuosity", "length"}
    assert "flow" not in panel._boxes
    assert "order" not in panel._boxes
    assert "resistance" not in panel._boxes


def test_branch_hover_panel_offers_flow_when_graph_has_it(make_napari_viewer):
    from haemolynx.gui._widget import settings_widget

    viewer = make_napari_viewer()
    settings_widget(napari_viewer=viewer)
    layer = _draw_hover(
        viewer,
        a_graph(flow_abs=1e-12, resistance=1e15, branch_order="A1"),
    )
    panel = _layer_controls(viewer, layer)._haemolynx_branch_hover
    assert panel.offered == ("flow", "order", "resistance", "tortuosity", "length")


def test_toggling_checkbox_rewrites_tooltip_strings(make_napari_viewer):
    from haemolynx.gui._widget import settings_widget

    viewer = make_napari_viewer()
    settings_widget(napari_viewer=viewer)
    layer = _draw_hover(viewer, a_graph(flow_abs=1.5e-12))
    panel = _layer_controls(viewer, layer)._haemolynx_branch_hover

    # Default selects every available metric.
    assert "branch flow" in str(layer.features["tooltip"][0])

    panel._boxes["flow"].setChecked(False)
    panel._boxes["tortuosity"].setChecked(False)
    text = str(layer.features["tooltip"][0])
    assert text == (
        "branchID: 0\n"
        f"branch length: {format_metric_value('length', 10.0)}"
    )
    assert "branch flow" not in text
    assert widget_mod._branch_hover_session_selected == ("length",)


def test_session_selection_persists_across_layer_rebuild(make_napari_viewer):
    from haemolynx.gui._widget import settings_widget

    viewer = make_napari_viewer()
    settings_widget(napari_viewer=viewer)
    layer = _draw_hover(viewer, a_graph())
    panel = _layer_controls(viewer, layer)._haemolynx_branch_hover
    panel._boxes["tortuosity"].setChecked(False)
    assert widget_mod._branch_hover_session_selected == ("length",)

    layer2 = _draw_hover(viewer, a_graph())
    panel2 = _layer_controls(viewer, layer2)._haemolynx_branch_hover
    assert panel2.selected == ("length",)
    assert str(layer2.features["tooltip"][0]) == (
        "branchID: 0\n"
        f"branch length: {format_metric_value('length', 10.0)}"
    )


def test_branch_hover_option_keys_are_not_passed_to_napari(make_napari_viewer):
    """Stashed availability keys must be stripped before add_points."""
    viewer = make_napari_viewer()
    layer = _draw_hover(viewer)
    # Survives on our metadata tag, not as a napari Points kwarg/attr.
    tag = layer.metadata["haemolynx"]
    assert tag["branch_hover_available"] == ("tortuosity", "length")
    assert not hasattr(layer, "branch_hover_available")


def test_branch_hover_update_with_stale_size_array_does_not_raise(make_napari_viewer):
    """Napari 0.9 rejects per-point size when len(size) != len(data)."""
    from haemolynx.gui._widget import _add_or_update, _apply_z_filter
    from haemolynx.gui.results import BRANCH_HOVER_POINT_SIZE, LayerSpec

    viewer = make_napari_viewer()
    layer = _draw_hover(viewer)
    full_count = len(layer.data)
    stale = np.full(full_count, BRANCH_HOVER_POINT_SIZE)
    layer.size = stale
    assert len(layer.size) == full_count

    # Fewer points, stale length-3909-style size array — must not raise.
    fewer = np.asarray(layer.data)[:2]
    features = {k: np.asarray(v)[:2] for k, v in layer.features.items()}
    _add_or_update(
        viewer,
        LayerSpec(
            kind="points",
            name=BRANCH_HOVER,
            data=fewer,
            features=features,
            options={"size": BRANCH_HOVER_POINT_SIZE},
        ),
    )
    updated = viewer.layers[BRANCH_HOVER]
    assert len(updated.data) == 2
    assert np.isscalar(updated.size) or len(np.asarray(updated.size)) == 2

    # Grow back in place with a size array left at the smaller count.
    layer = viewer.layers[BRANCH_HOVER]
    layer.size = np.full(len(layer.data), BRANCH_HOVER_POINT_SIZE)
    _draw_hover(viewer)
    assert len(viewer.layers[BRANCH_HOVER].data) == full_count


def test_branch_hover_z_filter_with_stale_size_array_does_not_raise(make_napari_viewer):
    """Z-filter shrink must sync Points size with the filtered row count."""
    from haemolynx.gui._widget import _apply_z_filter

    viewer = make_napari_viewer()
    layer = _draw_hover(viewer)
    full_count = len(layer.data)
    layer.size = np.full(full_count, 2.0)
    full_z = 30.0

    _apply_z_filter(viewer, 0.0, 15.0, z_extent=full_z)
    filtered = viewer.layers[BRANCH_HOVER]
    assert len(filtered.data) < full_count
    assert np.isscalar(filtered.size) or len(np.asarray(filtered.size)) == len(
        filtered.data
    )

    _apply_z_filter(viewer, 0.0, full_z, z_extent=full_z)
    restored = viewer.layers[BRANCH_HOVER]
    assert len(restored.data) == full_count
    assert np.isscalar(restored.size) or len(np.asarray(restored.size)) == full_count
