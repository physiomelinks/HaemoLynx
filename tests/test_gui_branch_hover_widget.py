"""Branch-hover layer registration and metrics panel in a real napari viewer."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402
from haemolynx.gui._widget import (  # noqa: E402
    OURS,
    _apply_layers,
    _branch_hover_mouse_move,
    _branch_hover_viewer_mouse_move,
    _layer_controls,
)
from haemolynx.gui.branch_hover import format_metric_value  # noqa: E402
from haemolynx.gui.results import (  # noqa: E402
    BRANCH_HOVER,
    VESSEL_LABELS,
    VESSEL_LABEL_POINT_SIZE,
    VESSEL_TUBES,
    VESSELS,
    LayerSpec,
    ResultLayers,
)
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
    return viewer.layers[VESSELS]


def _fake_event(position, dims_displayed=(0, 1, 2)):
    return SimpleNamespace(
        position=position,
        view_direction=None,
        dims_displayed=dims_displayed,
    )


def test_branch_hover_layer_registers_with_tooltip_features(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = _draw_hover(viewer)
    assert isinstance(layer, napari.layers.Vectors)
    assert "tooltip" in layer.features
    assert "branch_id" in layer.features
    assert str(layer.features["tooltip"][0]).startswith("branchID: ")
    assert _branch_hover_mouse_move in layer.mouse_move_callbacks
    assert _branch_hover_viewer_mouse_move in viewer.mouse_move_callbacks
    assert BRANCH_HOVER not in viewer.layers


def test_hovering_along_the_polyline_shows_tooltip(make_napari_viewer, monkeypatch):
    """A point on the centreline, not the old midpoint circle, shows the tip."""
    shown: list[str | None] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QToolTip.showText",
        lambda _pos, text, *a, **k: shown.append(str(text)),
    )
    monkeypatch.setattr(
        "qtpy.QtWidgets.QToolTip.hideText",
        lambda *a, **k: shown.append(None),
    )

    viewer = make_napari_viewer()
    layer = _draw_hover(viewer)
    assert layer.visible is False
    assert VESSEL_TUBES in viewer.layers
    assert viewer.layers[VESSEL_TUBES].visible is True
    # First vessel (0,0,0)->(10,0,0); midpoint was (5,0,0) with size 2.
    _branch_hover_mouse_move(layer, _fake_event((1.0, 0.0, 0.0)))
    assert shown
    assert shown[-1].startswith("branchID: 0")
    assert "branch length" in shown[-1]

    _branch_hover_mouse_move(layer, _fake_event((8.0, 0.0, 0.0)))
    assert shown[-1].startswith("branchID: 0")

    _branch_hover_viewer_mouse_move(viewer, _fake_event((15.0, 0.0, 0.0)))
    assert shown[-1].startswith("branchID: 1")

    _branch_hover_viewer_mouse_move(viewer, _fake_event((5.0, 50.0, 0.0)))
    assert shown[-1] is None


def test_legacy_midpoint_hover_circle_is_removed(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_points(
        np.array([[5.0, 0.0, 0.0]]),
        name=BRANCH_HOVER,
        metadata={OURS: {"kind": "points"}},
    )
    _draw_hover(viewer)
    assert BRANCH_HOVER not in viewer.layers
    assert isinstance(viewer.layers[VESSELS], napari.layers.Vectors)


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
    """Stashed availability keys must be stripped before add_vectors."""
    viewer = make_napari_viewer()
    layer = _draw_hover(viewer)
    tag = layer.metadata["haemolynx"]
    assert tag["branch_hover_available"] == ("tortuosity", "length")
    assert not hasattr(layer, "branch_hover_available")


def test_vessel_label_update_with_stale_size_array_does_not_raise(make_napari_viewer):
    """Napari 0.9 rejects per-point size when len(size) != len(data)."""
    from haemolynx.gui._widget import _add_or_update

    viewer = make_napari_viewer()
    _draw_hover(viewer)
    layer = viewer.layers[VESSEL_LABELS]
    full_count = len(layer.data)
    stale = np.full(full_count, VESSEL_LABEL_POINT_SIZE)
    layer.size = stale
    assert len(layer.size) == full_count

    fewer = np.asarray(layer.data)[:2]
    features = {k: np.asarray(v)[:2] for k, v in layer.features.items()}
    _add_or_update(
        viewer,
        LayerSpec(
            kind="points",
            name=VESSEL_LABELS,
            data=fewer,
            features=features,
            options={"size": VESSEL_LABEL_POINT_SIZE},
        ),
    )
    updated = viewer.layers[VESSEL_LABELS]
    assert len(updated.data) == 2
    assert np.isscalar(updated.size) or len(np.asarray(updated.size)) == 2

    layer = viewer.layers[VESSEL_LABELS]
    layer.size = np.full(len(layer.data), VESSEL_LABEL_POINT_SIZE)
    _draw_hover(viewer)
    assert len(viewer.layers[VESSEL_LABELS].data) == full_count


def test_vessel_label_z_filter_with_stale_size_array_does_not_raise(make_napari_viewer):
    """Z-filter shrink must sync Points size with the filtered row count."""
    from haemolynx.gui._widget import _apply_z_filter

    viewer = make_napari_viewer()
    _draw_hover(viewer)
    layer = viewer.layers[VESSEL_LABELS]
    full_count = len(layer.data)
    layer.size = np.full(full_count, 2.0)
    full_z = 30.0

    _apply_z_filter(viewer, 0.0, 15.0, z_extent=full_z)
    filtered = viewer.layers[VESSEL_LABELS]
    assert len(filtered.data) < full_count
    assert np.isscalar(filtered.size) or len(np.asarray(filtered.size)) == len(
        filtered.data
    )

    _apply_z_filter(viewer, 0.0, full_z, z_extent=full_z)
    restored = viewer.layers[VESSEL_LABELS]
    assert len(restored.data) == full_count
    assert np.isscalar(restored.size) or len(np.asarray(restored.size)) == full_count
