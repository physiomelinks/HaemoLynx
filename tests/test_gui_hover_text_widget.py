"""Every control HaemoLynx builds has hover text (``usability.6d``), checked by
scanning what is built rather than a list of strings kept by hand: the panel
with every Advanced button open, without a viewer and with one, and every
dock, window and layer control it opens beside itself. The scan is
:func:`hover_text_scan.controls_without_hover_text`; the Batch run window has
its own in ``test_gui_batch_run_widget.py``."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    _apply_layers,
    _GraphEditorWindow,
    _layer_controls,
    about_widget,
    settings_widget,
)
from haemolynx.gui.chrome_tooltips import STATISTICS_SECTION_TOOLTIPS  # noqa: E402
from haemolynx.gui.perturbation_editing import PERTURBATION_TYPES, UNCHOSEN  # noqa: E402
from haemolynx.gui.results import ResultLayers  # noqa: E402
from hover_text_scan import controls_without_hover_text  # noqa: E402

pytestmark = pytest.mark.gui


def _scan(roots: dict) -> dict[str, list[str]]:
    """Each root's controls with no hover text; a root with none is left out."""
    found = {name: controls_without_hover_text(root) for name, root in roots.items()}
    return {name: missing for name, missing in found.items() if missing}


def _beside_the_panel(panel) -> dict:
    """What the panel docks or keeps beside itself rather than in it: the
    view panel (docked when there is a viewer), the Statistics tab with a
    button for every measure and one with no entry of its own, each open,
    the log window and the arrow-size row it lends to a flow layer."""
    statistics = panel._haemolynx_statistics_panel
    sections = {key: {"A value": 1.0} for key in (*STATISTICS_SECTION_TOOLTIPS, "a_new_measure")}
    statistics.show({"mode": "fast", "csv": "out/run_statistics.csv", "sections": sections})
    assert set(statistics.buttons) == set(sections)
    for key in statistics.buttons:
        statistics.toggle(key, True)
    return {
        "panel": panel,
        "view panel": panel._haemolynx_view_panel,
        "statistics": statistics.widget,
        "log window": panel._haemolynx_log.native,
        "arrow size": panel._haemolynx_arrow_length_host,
    }


def test_a_path_rows_text_box_and_button_show_the_rows_hover_text(qapp):
    """Why the scan counts a FileEdit's parts as the row's: Qt passes hover
    over a part with no text of its own up to the row, so what the panel
    sets on the row (its setting's help, or why it is greyed) is what shows."""
    from magicgui.widgets import FileEdit
    from qtpy.QtCore import QEvent, QPoint
    from qtpy.QtGui import QHelpEvent
    from qtpy.QtWidgets import QApplication, QToolTip

    row = FileEdit(mode="r", name="a_path")
    row.tooltip = "The row's own hover text"
    row.native.show()
    try:
        for part in (row.line_edit.native, row.choose_btn.native):
            assert part.toolTip() == ""
            QToolTip.hideText()
            here = QPoint(2, 2)
            QApplication.sendEvent(part, QHelpEvent(QEvent.Type.ToolTip, here, part.mapToGlobal(here)))
            assert QToolTip.text() == "The row's own hover text"
        assert controls_without_hover_text(row.native) == []

        row.tooltip = ""
        assert controls_without_hover_text(row.native) == ["FileEdit 'a_path'"]
    finally:
        QToolTip.hideText()
        row.native.close()


def test_every_control_in_the_panel_without_a_viewer_has_hover_text(qapp):
    """Every row and button whether shown or not -- both optimisers' options
    and group lists among them -- and the Edit and About windows."""
    from qtpy.QtWidgets import QPushButton

    panel = settings_widget(napari_viewer=None)
    panel._haemolynx_expand_advanced()
    assert set(panel._haemolynx_optimise_group_checkboxes) and set(
        panel._haemolynx_optimise_fwhm_group_checkboxes
    )
    roots = _beside_the_panel(panel)
    roots["Edit window"] = _GraphEditorWindow(
        on_add=lambda: None, on_finish=lambda: None, on_delete=lambda: None,
        on_regenerate=lambda: None,
    ).native
    roots["About"] = about_widget()
    assert _scan(roots) == {}

    # The scan finds a control with none.
    bare = QPushButton("Bare", panel._haemolynx_view_panel)
    try:
        assert _scan(roots) == {"panel": ["QPushButton 'Bare'"], "view panel": ["QPushButton 'Bare'"]}
    finally:
        bare.setParent(None)


def _our_layer_controls(viewer) -> dict:
    """What HaemoLynx adds to each layer's own controls on the left."""
    from qtpy.QtWidgets import QWidget

    found = {}
    for layer in viewer.layers:
        controls = _layer_controls(viewer, layer)
        if controls is None:
            continue
        for attribute, addition in vars(controls).items():
            widget = addition if isinstance(addition, QWidget) else getattr(addition, "native", None)
            if attribute.startswith("_haemolynx_") and isinstance(widget, QWidget):
                found[f"{layer.name}: {attribute}"] = widget
    return found


def test_every_control_with_a_viewer_has_hover_text(make_napari_viewer, tmp_path):
    """What only a viewer brings: the open-layer row, the Boundaries tab's
    picking and box tools for every role, a perturbation of every type, the
    canvas's view buttons, a sweep's slider, and the controls added to the
    layers of a run -- a skeleton with a thick region, a mask the clean-up
    changed, a network and a sweep."""
    from haemolynx.pipeline import PerturbationRun
    from test_gui_results import a_graph, network
    from test_perturbation_stage import PRESSURE_SWEEP, _run

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((8, 8, 8)), name="stack")
    panel = settings_widget(napari_viewer=viewer)
    perturbations = panel._haemolynx_perturbations
    types = [kind for kind in PERTURBATION_TYPES if kind != UNCHOSEN]
    for index, kind in enumerate(types):
        perturbations.add()
        perturbations.choose_type(index, kind)
    panel._haemolynx_expand_advanced()

    one = np.zeros((4, 4, 4), dtype=bool)
    one[0, 0, 0] = True
    skeleton = one.copy()
    skeleton[1, 1, 1] = True
    _apply_layers(viewer, ResultLayers().stage_finished("skeletonise", SimpleNamespace(
        image=skeleton.astype(np.uint8), skeleton=skeleton, voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0), thick_vessel_mask=one, raw_segmented_image=one,
    )))
    _apply_layers(viewer, ResultLayers().stage_finished("build_network", network(a_graph())))
    sweep = _run(tmp_path, [PRESSURE_SWEEP]).results[0]
    _apply_layers(viewer, ResultLayers().stage_finished(
        "run_perturbations", PerturbationRun(results=[sweep], output_dir=tmp_path)))
    panel._haemolynx_after_layers_applied()

    assert len(perturbations.editors()) == len(types)
    assert panel._haemolynx_sweep_controls
    layer_controls = _our_layer_controls(viewer)
    added = {name.split(": ")[1] for name in layer_controls}
    assert {
        "_haemolynx_feature", "_haemolynx_colormap", "_haemolynx_scale",
        "_haemolynx_branch_hover", "_haemolynx_thick_thin", "_haemolynx_segmentation_cleanup",
        "_haemolynx_tube_quality", "_haemolynx_tube_diameter",
    } <= added
    roots = {
        **_beside_the_panel(panel),
        "view buttons": viewer._haemolynx_view_snap_buttons,
        **layer_controls,
    }
    assert _scan(roots) == {}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "The help window's directory and page have no hover text: words in that "
        "window are Harvey's to write (CLAUDE.md, Help menu), never an agent's"
    ),
)
def test_every_control_in_the_help_window_has_hover_text(qapp):
    panel = settings_widget(napari_viewer=None)
    panel._haemolynx_show_help()
    try:
        assert controls_without_hover_text(panel._haemolynx_help_window) == []
    finally:
        panel._haemolynx_help_window.close()
