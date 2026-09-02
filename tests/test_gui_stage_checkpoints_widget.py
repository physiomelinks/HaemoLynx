"""Revert-to-previous-stage button behaviour in a real napari panel.

The pure checkpoint decisions live in `test_gui_stage_checkpoints.py`. What is
left for here is the Qt wiring: the button exists on every tab but the first,
stays disabled until a prior stage has been checkpointed, restores layers, and
flips the skip toggles so the next Run loads the written graph pickle.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.results import (  # noqa: E402
    BOUNDARY_NODES,
    ResultLayers,
    VESSELS,
)
from haemolynx.gui.tabs import tab_titles as pipeline_tab_titles  # noqa: E402
from test_gui_results import a_graph, network  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    # Point the run's output at a writable temp tree so checkpoint pickles and
    # the resumed `{stem}_graph.pkl` have somewhere to land.
    rows = widget._haemolynx_rows()
    stem = "stack"
    rows["input_path"].value = tmp_path / f"{stem}.tif"
    rows["vtk_output_prefix"].value = tmp_path / "out" / stem
    (tmp_path / "out").mkdir()
    return widget, viewer, tmp_path


def _seed_run(widget, viewer, through: str = "assign_diameters") -> ResultLayers:
    """Pretend a run has finished through *through*, with checkpoints recorded."""
    results = ResultLayers()
    widget._haemolynx_view.results = results
    checkpoints = widget._haemolynx_checkpoints
    settings = widget._haemolynx_values()
    # Resolve paths the same way a run would, for the pickle locations.
    from haemolynx.pipeline import default_schema, resolve_settings

    resolved = resolve_settings(settings, schema=default_schema(), config_path=None)

    graph = a_graph(branch_order="A1", resistance=2.5)
    sequence = [
        (
            "skeletonise",
            SimpleNamespace(
                image=np.zeros((4, 4, 4), dtype=np.uint8),
                skeleton=np.zeros((4, 4, 4), dtype=bool),
                voxel_size_xyz=(1.0, 1.0, 1.0),
                voxel_size_zyx=(1.0, 1.0, 1.0),
            ),
        ),
        ("build_network", network(graph)),
        (
            "assign_boundaries",
            SimpleNamespace(
                inlet_nodes=[0],
                outlet_nodes=[3],
                arteriole_boundary_nodes=[],
                venule_boundary_nodes=[],
            ),
        ),
        ("assign_diameters", SimpleNamespace(graph=graph, results={})),
        ("build_haemodynamic_model", SimpleNamespace(graph=graph, results={})),
        (
            "solve",
            SimpleNamespace(
                pressure=np.asarray([1.0, 0.7, 0.3, 0.0]),
                node_list=[0, 1, 2, 3],
                equivalent_resistance=1.0,
            ),
        ),
    ]
    for stage, output in sequence:
        group = results.stage_finished(stage, output)
        checkpoints.record(stage, group, results, settings=resolved)
        from haemolynx.gui._widget import _apply_layers

        _apply_layers(viewer, group)
        if stage == through:
            break
    widget._haemolynx_refresh_revert()
    return results


# --- button presence and enablement ------------------------------------------


def test_every_tab_but_the_first_has_a_revert_button(panel):
    widget, _viewer, _tmp = panel
    titles = pipeline_tab_titles()
    buttons = widget._haemolynx_revert_buttons
    assert titles[0] not in buttons
    for title in titles[1:]:
        assert title in buttons
        assert buttons[title].text == "Revert to previous stage"
        assert buttons[title].enabled is False


def test_revert_stays_disabled_with_no_run(panel):
    widget, _viewer, _tmp = panel
    for button in widget._haemolynx_revert_buttons.values():
        assert button.enabled is False


def test_revert_enables_once_the_previous_stage_is_checkpointed(panel):
    widget, viewer, _tmp = panel
    _seed_run(widget, viewer, through="assign_boundaries")
    buttons = widget._haemolynx_revert_buttons
    assert buttons["5. Diameters"].enabled is True
    # Haemodynamics wants the diameters checkpoint, which is not there yet.
    assert buttons["6. Haemodynamics"].enabled is False


# --- restore behaviour -------------------------------------------------------


def test_revert_restores_previous_tab_layers_and_selects_that_tab(panel):
    widget, viewer, _tmp = panel
    _seed_run(widget, viewer, through="assign_diameters")
    assert BOUNDARY_NODES in viewer.layers
    assert VESSELS in viewer.layers

    widget._haemolynx_revert("5. Diameters")

    assert BOUNDARY_NODES in viewer.layers
    assert VESSELS in viewer.layers
    tabs = widget._haemolynx_tabs
    assert tabs.tabText(tabs.currentIndex()) == "4. Boundaries"
    report = widget._haemolynx_report()
    assert "4. Boundaries" in report or "Boundaries" in report


def test_revert_writes_graph_pkl_and_turns_off_rebuild_toggles(panel):
    widget, viewer, tmp_path = panel
    _seed_run(widget, viewer, through="assign_diameters")
    rows = widget._haemolynx_rows()
    rows["do_skeletonize"].value = True
    rows["do_graph_building"].value = True

    widget._haemolynx_revert("5. Diameters")

    assert rows["do_skeletonize"].value is False
    assert rows["do_graph_building"].value is False
    report = widget._haemolynx_report()
    assert "do_skeletonize" in report
    # resolve_settings may absolutise vtk_output_prefix; find the resume pickle
    # wherever the panel actually wrote it.
    written = list(tmp_path.rglob("*_graph.pkl"))
    assert written, f"no graph.pkl under {tmp_path}; report was: {report}"
    import pickle

    with written[0].open("rb") as handle:
        restored = pickle.load(handle)
    assert isinstance(restored, nx.MultiGraph)
    assert restored.number_of_nodes() == 4


def test_revert_with_nothing_saved_says_so_and_does_not_crash(panel):
    widget, _viewer, _tmp = panel
    widget._haemolynx_revert("5. Diameters")
    assert "Nothing to restore" in widget._haemolynx_report()


def test_clear_layers_forgets_checkpoints_and_disables_revert(panel):
    widget, viewer, _tmp = panel
    _seed_run(widget, viewer, through="assign_boundaries")
    assert widget._haemolynx_revert_buttons["5. Diameters"].enabled is True

    widget._haemolynx_clear()

    assert widget._haemolynx_checkpoints.stages == ()
    assert widget._haemolynx_revert_buttons["5. Diameters"].enabled is False
