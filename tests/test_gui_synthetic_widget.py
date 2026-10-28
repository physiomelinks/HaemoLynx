"""The real panel under a synthetic network (``network_source=synthetic``):
the Input tab swaps its rows, the open-layer row goes, the Boundaries role
tabs grey with the generator's note, and the template row opens a
``.haemorun`` dialogue."""
from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.boundary_picking import ROLES, SYNTHETIC_CHOOSES_NOTE  # noqa: E402
from haemolynx.gui.form import PATH_FILTERS  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((20, 40, 40), dtype=np.uint8), name="stack")
    widget = settings_widget(napari_viewer=viewer)
    return widget, viewer


def _process():
    from qtpy.QtWidgets import QApplication

    QApplication.processEvents()


def test_the_input_tab_swaps_its_rows_and_greys_the_role_tabs(panel):
    widget, _viewer = panel
    widget.show()
    rows = widget._haemolynx_rows()
    tabs = widget._haemolynx_boundaries.state.tabs

    assert rows["network_source"].value == "segmented_image"
    assert rows["input_path"].visible and rows["raw_image_path"].visible
    assert not rows["synthetic_template_path"].visible
    assert not rows["synthetic_template_source"].visible
    # Inlet, Outlet, Arteriole, Venule are manual by default; the two
    # large-vessel tabs wait on assign_large_vessel_branch_orders.
    manual = [i for i in range(tabs.count()) if not ROLES[i].startswith("large_vessel")]
    assert all(tabs.isTabEnabled(i) for i in manual)

    rows["network_source"].value = "synthetic"
    _process()
    assert rows["synthetic_template_path"].visible
    assert rows["synthetic_template_source"].visible
    for name in ("input_path", "raw_image_path", "image_axis_order", "use_memmap_loading", "segmentation_cleanup", "crop_on_import"):
        assert not rows[name].visible, name
    assert not rows["automated_vessel_assignment"].visible
    for index in range(tabs.count()):
        assert not tabs.isTabEnabled(index), ROLES[index]
        assert tabs.tabToolTip(index) == SYNTHETIC_CHOOSES_NOTE
    assert widget._haemolynx_values()["network_source"] == "synthetic"

    rows["network_source"].value = "segmented_image"
    _process()
    assert rows["input_path"].visible
    assert not rows["synthetic_template_path"].visible
    assert all(tabs.isTabEnabled(i) for i in manual)


def test_the_template_row_is_a_haemorun_file_picker(panel):
    widget, _viewer = panel
    rows = widget._haemolynx_rows()
    picker = rows["synthetic_template_path"]
    assert picker.mode.value == "r" if hasattr(picker.mode, "value") else str(picker.mode).endswith("r")
    assert picker.filter == PATH_FILTERS["synthetic_template_path"]
    assert "*.haemobatch" in picker.filter


def test_the_advanced_synthetic_rows_sit_behind_their_own_button(panel):
    widget, _viewer = panel
    widget.show()
    rows = widget._haemolynx_rows()
    rows["network_source"].value = "synthetic"
    _process()
    assert not rows["synthetic_seed"].visible
    widget._haemolynx_expand_advanced()
    _process()
    for name in (
        "synthetic_seed", "synthetic_scale", "synthetic_scale_axes", "synthetic_extension_arteriole_density",
        "synthetic_extension_venule_density", "synthetic_interior_dead_ends",
        "synthetic_open_ends", "synthetic_bed_model",
    ):
        assert rows[name].visible, name
    assert rows["synthetic_seed"].value == 20240917


def test_picking_a_template_sets_the_segmentation_source_to_its_basis(panel, tmp_path):
    """Harvey, 2026-10-24: the row follows the template's diameter basis when
    the template is picked; a config loaded afterwards says what it says."""
    from synthetic_fixtures import write_fixture_run

    template = write_fixture_run(tmp_path, settings={"segmentation_source": "endothelial"}, name="claudin.haemorun")
    widget, _viewer = panel
    widget.show()
    rows = widget._haemolynx_rows()
    assert rows["segmentation_source"].value == "plasma_column"
    rows["network_source"].value = "synthetic"
    rows["synthetic_template_path"].value = template
    _process()
    assert rows["segmentation_source"].value == "endothelial"
    assert widget._haemolynx_values()["segmentation_source"] == "endothelial"

    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            (
                "network_source: synthetic",
                f"synthetic_template_path: {template.as_posix()}",
                "segmentation_source: plasma_column",
                "",
            )
        ),
        encoding="utf-8",
    )
    widget._haemolynx_load_config(config)
    _process()
    assert rows["segmentation_source"].value == "plasma_column"


def _wheel(bar, notch_fraction: float):
    from qtpy.QtCore import QPoint, QPointF, Qt
    from qtpy.QtGui import QWheelEvent

    centre = QPointF(bar.rect().center())
    return QWheelEvent(
        centre, bar.mapToGlobal(centre), QPoint(0, 0), QPoint(0, int(120 * notch_fraction)),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
    )


def test_a_fine_wheel_over_the_greyed_role_tabs_does_not_hang_the_panel(panel):
    """Regression (Harvey, 2026-10-24: opening the Boundaries tab under a
    synthetic network hung napari in QTabBar::wheelEvent). From a disabled
    current tab, a wheel turned less than a notch makes Qt look for the next
    enabled tab with a step of nought, forever; the panel swallows it."""
    from qtpy.QtWidgets import QApplication

    widget, _viewer = panel
    widget.show()
    rows = widget._haemolynx_rows()
    state = widget._haemolynx_boundaries.state
    tabs = state.tabs
    bar = tabs.tabBar()
    guard = state.tab_wheel_guard
    rows["network_source"].value = "synthetic"
    _process()
    assert not any(tabs.isTabEnabled(i) for i in range(tabs.count()))
    fine = _wheel(bar, 0.1)
    assert guard.eventFilter(bar, fine) is True
    # The guard is in place (asserted above), so delivering the event returns.
    before = tabs.currentIndex()
    QApplication.sendEvent(bar, fine)
    assert tabs.currentIndex() == before
    # Over an enabled tab the wheel is Qt's again.
    rows["network_source"].value = "segmented_image"
    _process()
    assert tabs.isTabEnabled(tabs.currentIndex())
    assert guard.eventFilter(bar, _wheel(bar, 1.0)) is False
