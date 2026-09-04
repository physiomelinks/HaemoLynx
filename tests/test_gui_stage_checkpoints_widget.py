"""Run-from-this-stage button behaviour in a real napari panel.

The pure checkpoint decisions live in `test_gui_stage_checkpoints.py`. What is
left for here is the Qt wiring: the button exists on every tab but the first,
stays disabled until a prior stage has been checkpointed, restores earlier
layers, flips the skip toggles so the run loads the written graph pickle, and
starts the pipeline at this tab.
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
        assert buttons[title].text == "Run from this stage"
        assert buttons[title].enabled is False


def test_revert_sits_below_show_each_topology_step(panel):
    """Revert chrome is directly under the show-steps checkbox, centered.

    Layout index in the panel column is the contract: view controls (which own
    "Show each topology step") come before the per-tab Revert stack. Geometry
    then confirms the active tab's button is below that checkbox and centered
    on the panel once Qt has laid out.
    """
    from qtpy.QtWidgets import QApplication

    widget, _viewer, _tmp = panel
    layout = widget.layout()
    view_controls = widget._haemolynx_view_controls.native
    show_steps = widget._haemolynx_show_steps.native
    revert_stack = widget._haemolynx_revert_stack

    assert view_controls.objectName() == "haemolynx_view_controls"
    assert show_steps.objectName() == "haemolynx_show_steps"
    assert revert_stack.objectName() == "haemolynx_revert_stack"

    view_index = layout.indexOf(view_controls)
    revert_index = layout.indexOf(revert_stack)
    assert view_index >= 0
    assert revert_index >= 0
    assert view_index < revert_index, (
        "Revert stack must sit below the show-results / show-topology-steps row"
    )
    # Nothing else between them: Revert is immediately under that chrome.
    assert revert_index == view_index + 1

    ancestor = show_steps.parentWidget()
    while ancestor is not None and ancestor is not view_controls:
        ancestor = ancestor.parentWidget()
    assert ancestor is view_controls, (
        "'Show each topology step' must live inside the view-controls chrome"
    )

    titles = pipeline_tab_titles()
    assert titles[1] in widget._haemolynx_revert_buttons
    tabs = widget._haemolynx_tabs
    tabs.setCurrentIndex(1)
    QApplication.processEvents()
    assert revert_stack.currentIndex() == 1

    widget.resize(420, 900)
    widget.show()
    QApplication.processEvents()

    revert = widget._haemolynx_revert_buttons[titles[1]].native
    assert revert.objectName() == "haemolynx_revert"
    assert revert.isVisible()
    steps_bottom = show_steps.mapTo(widget, show_steps.rect().bottomLeft()).y()
    revert_top = revert.mapTo(widget, revert.rect().topLeft()).y()
    assert steps_bottom <= revert_top, (
        f"Revert top ({revert_top}) must be at or below show-steps bottom "
        f"({steps_bottom})"
    )
    panel_mid = widget.rect().center().x()
    revert_mid = revert.mapTo(widget, revert.rect().center()).x()
    assert abs(panel_mid - revert_mid) <= 8, (
        f"Revert should be centered on the panel (mid={panel_mid}, "
        f"button mid={revert_mid})"
    )


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


def test_revert_restores_previous_tab_layers_and_stays_on_this_tab(panel):
    widget, viewer, _tmp = panel
    _seed_run(widget, viewer, through="assign_diameters")
    assert BOUNDARY_NODES in viewer.layers
    assert VESSELS in viewer.layers

    widget._haemolynx_revert("5. Diameters")

    assert BOUNDARY_NODES in viewer.layers
    assert VESSELS in viewer.layers
    tabs = widget._haemolynx_tabs
    assert tabs.tabText(tabs.currentIndex()) == "5. Diameters"
    report = widget._haemolynx_report()
    assert "Running from 5. Diameters" in report


def test_revert_writes_graph_pkl_and_turns_off_rebuild_toggles(panel):
    widget, viewer, tmp_path = panel
    _seed_run(widget, viewer, through="assign_diameters")
    rows = widget._haemolynx_rows()
    rows["do_skeletonize"].value = True
    rows["do_graph_building"].value = True

    widget._haemolynx_revert("5. Diameters")

    assert rows["do_skeletonize"].value is False
    assert rows["do_graph_building"].value is False
    assert rows["do_fwhm_measurement"].value is True
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


def test_revert_from_haemodynamics_turns_off_fwhm_remeasurement(panel):
    """Continuing from Haemodynamics must keep diameters already on the graph."""
    widget, viewer, _tmp = panel
    _seed_run(widget, viewer, through="assign_diameters")
    rows = widget._haemolynx_rows()
    rows["use_fwhm_edge_diameters"].value = True
    rows["do_fwhm_measurement"].value = True

    widget._haemolynx_revert("6. Haemodynamics")

    assert rows["do_fwhm_measurement"].value is False
    assert rows["do_graph_building"].value is False
    tabs = widget._haemolynx_tabs
    assert tabs.tabText(tabs.currentIndex()) == "6. Haemodynamics"


def test_revert_with_nothing_saved_says_so_and_does_not_crash(panel):
    widget, _viewer, _tmp = panel
    widget._haemolynx_revert("5. Diameters")

    assert "Nothing ready to run from" in widget._haemolynx_report() or (
        "Nothing to restore" in widget._haemolynx_report()
    )


def test_clear_layers_forgets_checkpoints_and_disables_revert(panel):
    widget, viewer, tmp_path = panel
    _seed_run(widget, viewer, through="assign_boundaries")
    assert widget._haemolynx_revert_buttons["5. Diameters"].enabled is True
    rows = widget._haemolynx_rows()
    rows["do_skeletonize"].value = True
    rows["do_graph_building"].value = True

    widget._haemolynx_revert("5. Diameters")

    assert rows["do_skeletonize"].value is False
    assert rows["do_graph_building"].value is False
    assert list(tmp_path.rglob("*_graph.pkl"))

    widget._haemolynx_clear()

    assert widget._haemolynx_checkpoints.stages == ()
    assert widget._haemolynx_revert_buttons["5. Diameters"].enabled is False
    assert rows["do_skeletonize"].value is True
    assert rows["do_graph_building"].value is True
    assert not list(tmp_path.rglob("*_graph.pkl"))
    assert not list(tmp_path.rglob("*_checkpoint_*.pkl"))
    report = widget._haemolynx_report()
    assert "Discarded cached" in report
    assert "Restored skeletonize" in report
    assert widget._haemolynx_run_button.enabled is True


def test_revert_from_every_later_tab_selects_the_restored_tab(panel):
    """For every tab K with a predecessor, run-from on K stays on K."""
    from qtpy.QtWidgets import QApplication

    from haemolynx.gui.stage_checkpoints import revert_target_stage

    widget, viewer, _tmp = panel
    titles = pipeline_tab_titles()
    tabs = widget._haemolynx_tabs

    for index in range(1, len(titles)):
        current = titles[index]
        if current not in widget._haemolynx_revert_buttons:
            continue
        _seed_run(widget, viewer, through="solve")
        target = revert_target_stage(current)
        if target is None or not widget._haemolynx_checkpoints.has(target):
            continue

        tabs.setCurrentIndex(index)
        widget._haemolynx_revert(current)
        QApplication.processEvents()

        assert tabs.tabText(tabs.currentIndex()) == current, (
            f"run-from on {current!r} should stay on {current!r}, "
            f"got {tabs.tabText(tabs.currentIndex())!r}"
        )


def test_clicking_run_from_on_boundaries_stays_on_boundaries(panel, monkeypatch):
    """Native click stays on this tab and starts a run from assign_boundaries."""
    from qtpy.QtWidgets import QApplication

    from haemolynx.gui import _widget as widget_mod
    from haemolynx.parsers.checks import CheckReport

    started = []

    def fake_background(*_args, **kwargs):
        started.append(kwargs)
        return None

    monkeypatch.setattr(widget_mod, "_run_in_background", fake_background)
    monkeypatch.setattr(widget_mod, "preflight", lambda *_a, **_k: CheckReport())

    widget, viewer, _tmp = panel
    _seed_run(widget, viewer, through="assign_diameters")
    tabs = widget._haemolynx_tabs
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    tabs.setCurrentIndex(titles.index("4. Boundaries"))

    widget._haemolynx_revert_buttons["4. Boundaries"].native.click()
    QApplication.processEvents()

    assert tabs.tabText(tabs.currentIndex()) == "4. Boundaries"
    rows = widget._haemolynx_rows()
    assert rows["do_skeletonize"].value is False
    assert rows["do_graph_building"].value is False
    assert started
    assert started[0].get("start_from") == "assign_boundaries"
    assert started[0].get("resume") is not None


def test_after_run_from_boundaries_cached_artefacts_pass_preflight(panel):
    """Preparing Boundaries writes the Graph pickle so a run can load it."""
    from haemolynx.pipeline import default_schema, preflight, resolve_settings
    from haemolynx.pipeline.checks import check_cached_artefacts

    widget, viewer, tmp_path = panel
    _seed_run(widget, viewer, through="assign_diameters")
    rows = widget._haemolynx_rows()
    rows["do_skeletonize"].value = True
    rows["do_graph_building"].value = True

    widget._haemolynx_revert("4. Boundaries")

    assert rows["do_skeletonize"].value is False
    assert rows["do_graph_building"].value is False
    settings = resolve_settings(
        widget._haemolynx_values(), schema=default_schema(), config_path=None
    )
    cached = check_cached_artefacts(settings)
    assert cached.ok, cached.errors
    # Full preflight may still fail on missing input_path files in the temp
    # form; the resume-specific failure mode was the cached-artefact check.
    assert list(tmp_path.rglob("*_skeleton.npy")), "skeleton.npy missing after revert"
    assert list(tmp_path.rglob("*_graph.pkl")), "graph.pkl missing after revert"


def test_save_and_load_run_buttons_sit_under_run_pipeline_on_the_right(panel):
    from qtpy.QtWidgets import QApplication

    widget, _viewer, _tmp = panel
    layout = widget.layout()
    run_row = widget._haemolynx_run_file_row
    buttons_native = widget._haemolynx_run_button.native.parentWidget()
    while buttons_native is not None and layout.indexOf(buttons_native) < 0:
        buttons_native = buttons_native.parentWidget()
    assert run_row.objectName() == "haemolynx_run_file_row"
    assert widget._haemolynx_save_run_button.text == "Save run..."
    assert widget._haemolynx_load_run_button.text == "Load run..."
    assert layout.indexOf(run_row) == layout.indexOf(buttons_native) + 1

    widget.resize(420, 900)
    widget.show()
    QApplication.processEvents()
    panel_right = widget.rect().right()
    load_right = widget._haemolynx_load_run_button.native.mapTo(
        widget, widget._haemolynx_load_run_button.native.rect().topRight()
    ).x()
    assert abs(panel_right - load_right) <= 24, (
        f"Load run should sit on the right (panel right={panel_right}, "
        f"button right={load_right})"
    )


def test_save_run_with_nothing_recorded_says_so(panel):
    widget, _viewer, tmp_path = panel
    dest = tmp_path / "empty.haemorun"
    assert widget._haemolynx_save_run(dest) is False
    assert "Nothing to save" in widget._haemolynx_report()
    assert not dest.exists()


def test_save_then_load_run_restores_layers_and_checkpoints(panel):
    widget, viewer, tmp_path = panel
    _seed_run(widget, viewer, through="assign_diameters")
    assert VESSELS in viewer.layers
    dest = tmp_path / "session.haemorun"

    assert widget._haemolynx_save_run(dest) is True
    assert dest.is_file() or dest.with_suffix(".haemorun").is_file()
    saved = dest if dest.is_file() else dest.with_suffix(".haemorun")

    widget._haemolynx_clear()
    assert VESSELS not in viewer.layers
    assert widget._haemolynx_checkpoints.stages == ()

    assert widget._haemolynx_load_run(saved) is True
    assert VESSELS in viewer.layers
    assert BOUNDARY_NODES in viewer.layers
    assert "assign_diameters" in widget._haemolynx_checkpoints.stages
    assert widget._haemolynx_view.results is not None
    assert widget._haemolynx_view.results._graph is not None
    assert widget._haemolynx_revert_buttons["6. Haemodynamics"].enabled is True
    assert "Loaded run" in widget._haemolynx_report()

