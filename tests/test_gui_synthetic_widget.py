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
from haemolynx.gui.run_snapshot import SAVE_FILTER  # noqa: E402

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
    assert picker.filter == SAVE_FILTER


def test_the_advanced_synthetic_rows_sit_behind_their_own_button(panel):
    widget, _viewer = panel
    widget.show()
    rows = widget._haemolynx_rows()
    rows["network_source"].value = "synthetic"
    _process()
    assert not rows["synthetic_seed"].visible
    widget._haemolynx_expand_advanced()
    _process()
    for name in ("synthetic_seed", "synthetic_scale", "synthetic_interior_dead_ends"):
        assert rows[name].visible, name
    assert rows["synthetic_seed"].value == 20240917
