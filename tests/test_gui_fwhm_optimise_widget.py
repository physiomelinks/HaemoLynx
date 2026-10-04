"""The "Optimise FWHM settings" button, built for real, with a live
QApplication -- mirrors tests/test_gui_optimise_settings_widget.py's own
structure and rationale for the Input tab's "Optimise settings" button.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402
from haemolynx.gui._widget import settings_widget  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(qapp):
    return settings_widget(napari_viewer=None)


def _fake_results(graph, voxel_size_zyx=(1.0, 1.0, 1.0)):
    return SimpleNamespace(_graph=graph, _voxel_size_zyx=voxel_size_zyx)


def _tiny_graph():
    import networkx as nx
    import numpy as np

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, 10.0]))
    G.add_edge(0, 1, length=10.0, voxels=[(0.0, 0.0, float(x)) for x in range(11)])
    return G


# --- the button exists and is wired -----------------------------------------


def test_optimise_fwhm_button_exists_with_a_tooltip(panel):
    button = panel._haemolynx_optimise_fwhm_button
    assert button.text == "Optimise FWHM settings"
    assert button.tooltip.strip()


def test_optimise_fwhm_button_sits_right_after_the_raw_tiff_row(panel):
    """Per the feature request: 'under the master checkbox and input file
    line for FWHM, but before the other settings'."""
    container = panel._haemolynx_diameters_settings
    widgets = list(container)
    rows = panel._haemolynx_rows()
    raw_tiff_index = widgets.index(rows["fwhm_raw_tiff_path"])
    button_index = widgets.index(panel._haemolynx_optimise_fwhm_button)
    assert button_index == raw_tiff_index + 1
    # And before the next FWHM setting row.
    next_row_index = widgets.index(rows["fwhm_sample_spacing_along_edge_um"])
    assert button_index < next_row_index


def test_optimise_fwhm_button_hidden_until_fwhm_measurement_is_turned_on(panel):
    """A nested widget's own ``.visible`` only reflects reality once its
    top-level ancestor is shown *and* its tab is the active one (an inactive
    QTabWidget page stays hidden regardless) -- see
    test_gui_optimise_settings_widget.py's own
    test_ticking_choose_groups_reveals_the_checkbox_list for the same
    caveat; that one only needed `.show()` because its widget lives on the
    already-active "1. Input" tab, while this button lives on "5. Diameters"."""
    from haemolynx.gui.tabs import tab_titles

    panel.show()
    diameters_index = list(tab_titles()).index("5. Diameters")
    panel._haemolynx_tabs.setCurrentIndex(diameters_index)
    rows = panel._haemolynx_rows()
    rows["use_fwhm_edge_diameters"].value = False
    assert panel._haemolynx_optimise_fwhm_button.visible is False

    rows["use_fwhm_edge_diameters"].value = True
    assert panel._haemolynx_optimise_fwhm_button.visible is True

    rows["use_fwhm_edge_diameters"].value = False
    assert panel._haemolynx_optimise_fwhm_button.visible is False


# --- guards ------------------------------------------------------------------


def test_optimise_fwhm_settings_refuses_without_a_graph(panel, monkeypatch):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *a, **k: started.append((a, k)),
    )
    panel._haemolynx_optimise_fwhm_settings()
    assert not started
    assert "run the pipeline through at least Graph first" in panel._haemolynx_report()


def test_optimise_fwhm_settings_refuses_without_a_raw_path(panel, monkeypatch):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *a, **k: started.append((a, k)),
    )
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    panel._haemolynx_optimise_fwhm_settings()
    assert not started
    assert "Choose a raw FWHM image" in panel._haemolynx_report()


def test_optimise_fwhm_settings_refuses_while_a_run_is_already_going(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *a, **k: started.append((a, k)),
    )
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    panel._haemolynx_rows()["fwhm_raw_tiff_path"].value = raw_file

    panel._haemolynx_run_state.start(worker=None, cancel_flag={"cancelled": False})
    try:
        panel._haemolynx_optimise_fwhm_settings()
    finally:
        panel._haemolynx_run_state.stopped()
    assert not started
    assert "already going" in panel._haemolynx_report()


# --- wiring: the button starts a background worker with the right arguments --


def test_optimise_fwhm_settings_starts_the_background_worker(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *args, **kwargs: started.append((args, kwargs)),
    )
    graph = _tiny_graph()
    panel._haemolynx_view.results = _fake_results(graph, voxel_size_zyx=(2.0, 1.0, 1.0))
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    panel._haemolynx_rows()["fwhm_raw_tiff_path"].value = raw_file

    panel._haemolynx_optimise_fwhm_settings()

    assert len(started) == 1
    args, kwargs = started[0]
    passed_graph, voxel_size_zyx, settings, schema, passed_rows, report, button, bars = args
    assert passed_graph is not graph  # a copy, never the caller's own graph
    assert passed_graph.number_of_edges() == graph.number_of_edges()
    assert voxel_size_zyx == (2.0, 1.0, 1.0)
    assert passed_rows is panel._haemolynx_rows()
    assert button is panel._haemolynx_optimise_fwhm_button
    assert bars is panel._haemolynx_optimise_fwhm_bars
    assert kwargs["apply_prerequisites"] is not None
    assert kwargs["run_state"] is panel._haemolynx_run_state
    assert kwargs["groups"] is None
    assert kwargs["sample_edge_count"] is None  # Auto
    assert kwargs["max_passes"] == 1
    assert kwargs["propose"] == panel._haemolynx_optimise_fwhm_review.propose


@pytest.mark.parametrize("skeletonised", [True, False])
def test_optimise_fwhm_settings_hands_the_worker_the_runs_segmented_image(
    panel, monkeypatch, tmp_path, skeletonised
):
    """The segmented image the skeletonise stage kept is what the optimiser
    places decoys and reads the mask's own widths from; a run with no such
    checkpoint (a loaded one) optimises without them."""
    import numpy as np

    from haemolynx.gui.stage_checkpoints import StageCheckpoint

    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *args, **kwargs: started.append(kwargs),
    )
    mask = np.ones((3, 3, 11), dtype=bool)
    if skeletonised:
        panel._haemolynx_checkpoints.replace_all(
            [StageCheckpoint(stage="skeletonise", title="2. Skeletonise", group=None,
                             output=SimpleNamespace(image=mask))]
        )
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    panel._haemolynx_rows()["fwhm_raw_tiff_path"].value = raw_file

    panel._haemolynx_optimise_fwhm_settings()

    assert (started[0]["vessel_mask"] is mask) if skeletonised else started[0]["vessel_mask"] is None


def test_the_settings_the_fwhm_optimiser_reads_are_real_settings():
    from haemolynx.pipeline.schema import SCHEMA

    assert set(widget_mod._FWHM_OPTIMISER_READS) <= {setting.name for setting in SCHEMA}


def test_clicking_the_native_fwhm_button_triggers_optimisation(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *args, **kwargs: started.append(args),
    )
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    panel._haemolynx_rows()["fwhm_raw_tiff_path"].value = raw_file
    panel._haemolynx_rows()["use_fwhm_edge_diameters"].value = True

    panel._haemolynx_optimise_fwhm_button.native.click()

    assert len(started) == 1


# --- clearing resets the button and bars --------------------------------------


def test_on_clear_resets_the_fwhm_optimise_button_and_bars(panel):
    panel._haemolynx_optimise_fwhm_button.enabled = False
    panel._haemolynx_run_state.start(worker=None, cancel_flag={"cancelled": False})

    panel._haemolynx_clear(ask=False)

    assert panel._haemolynx_optimise_fwhm_button.enabled is True


# --- shared cancellation/progress-bridge helper (also used by
# _run_optimisation_in_background) -------------------------------------------


def test_cancellable_progress_bridge_still_ours_reflects_run_state_identity(qapp):
    """Regression: _run_fwhm_optimisation_in_background used to build this
    cancel_flag/still_ours/watched plumbing inline, copy-pasted from
    _run_optimisation_in_background -- both now share
    _cancellable_progress_bridge."""
    from haemolynx.gui.run_state import RunState

    run_state = RunState()
    bars = SimpleNamespace(show_event=lambda event: None)
    cancel_flag, still_ours, _watched = widget_mod._cancellable_progress_bridge(run_state, bars)

    assert cancel_flag == {"cancelled": False}
    assert still_ours() is False  # no run has claimed run_state yet

    run_state.start(worker=None, cancel_flag=cancel_flag)
    assert still_ours() is True

    other_cancel_flag = {"cancelled": False}
    run_state.start(worker=None, cancel_flag=other_cancel_flag)
    assert still_ours() is False  # superseded by a different run


def test_cancellable_progress_bridge_watched_raises_once_cancelled(qapp):
    from haemolynx.gui.run_state import RunCancelled, RunState

    run_state = RunState()
    bars = SimpleNamespace(show_event=lambda event: None)
    cancel_flag, _still_ours, watched = widget_mod._cancellable_progress_bridge(run_state, bars)
    run_state.start(worker=None, cancel_flag=cancel_flag)

    watched(SimpleNamespace(kind="group_started"))  # does not raise before cancel

    cancel_flag["cancelled"] = True
    with pytest.raises(RunCancelled):
        watched(SimpleNamespace(kind="group_started"))


def test_start_optimisation_worker_wires_up_button_bars_and_report(qapp):
    """Regression: the final worker-wiring block (button/bars/report/
    run_state.start) used to be duplicated verbatim between
    _run_optimisation_in_background and _run_fwhm_optimisation_in_background."""
    from haemolynx.gui.run_state import RunState

    run_state = RunState()
    bars = SimpleNamespace(
        show_event=lambda event: None,
        start=lambda: started_calls.append("start"),
    )
    started_calls: list[str] = []
    button = SimpleNamespace(enabled=True)
    report = SimpleNamespace(value="")
    cancel_flag, still_ours, _watched = widget_mod._cancellable_progress_bridge(run_state, bars)

    class _FakeWorker:
        def __init__(self):
            self.returned = SimpleNamespace(connect=lambda fn: None)
            self.finished = SimpleNamespace(connect=lambda fn: None)
            self.started = False

        def start(self):
            self.started = True

    fake_worker = _FakeWorker()

    def run_thread_worker(*, _connect, _start_thread):
        assert _start_thread is False
        assert "errored" in _connect
        return fake_worker

    result = widget_mod._start_optimisation_worker(
        run_thread_worker,
        run_state=run_state,
        cancel_flag=cancel_flag,
        still_ours=still_ours,
        button=button,
        bars=bars,
        report=report,
        finished=lambda payload: None,
        failed=lambda error: None,
        starting_message="Optimising for a test...",
    )

    assert result is fake_worker
    assert fake_worker.started is True
    assert button.enabled is False
    assert report.value == "Optimising for a test..."
    assert started_calls == ["start"]
    assert run_state.running is True


# --- the FWHM optimiser's own options, and its review -----------------------------


def test_the_fwhm_options_are_passed_through(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod, "_run_fwhm_optimisation_in_background", lambda *a, **k: started.append((a, k))
    )
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    panel._haemolynx_rows()["fwhm_raw_tiff_path"].value = raw_file
    panel._haemolynx_optimise_fwhm_choose_groups.value = True
    for name, box in panel._haemolynx_optimise_fwhm_group_checkboxes.items():
        box.value = name in ("exclusion_zones", "rejection_gates")
    panel._haemolynx_optimise_fwhm_sample.value = "50 vessels"
    panel._haemolynx_optimise_fwhm_passes.value = 2

    panel._haemolynx_optimise_fwhm_settings()

    kwargs = started[0][1]
    assert kwargs["groups"] == ("exclusion_zones", "rejection_gates")
    assert kwargs["sample_edge_count"] == 50
    assert kwargs["max_passes"] == 2


def test_the_fwhm_options_and_review_show_only_with_fwhm_measurement(panel):
    from haemolynx.gui.tabs import tab_titles

    panel.show()
    panel._haemolynx_tabs.setCurrentIndex(list(tab_titles()).index("5. Diameters"))
    rows = panel._haemolynx_rows()
    box = panel._haemolynx_optimise_fwhm_box

    rows["use_fwhm_edge_diameters"].value = False
    assert box.visible is False
    rows["use_fwhm_edge_diameters"].value = True
    assert box.visible is True
    assert panel._haemolynx_optimise_fwhm_options_container.visible is False

    panel._haemolynx_optimise_fwhm_options.value = True
    assert panel._haemolynx_optimise_fwhm_options_container.visible is True


def test_the_fwhm_options_sit_right_under_the_button(panel):
    widgets = list(panel._haemolynx_diameters_settings)
    button_index = widgets.index(panel._haemolynx_optimise_fwhm_button)
    assert widgets.index(panel._haemolynx_optimise_fwhm_box) == button_index + 1


def test_an_fwhm_proposal_applies_only_on_apply(panel):
    from types import SimpleNamespace

    review = panel._haemolynx_optimise_fwhm_review
    applied = []
    message = review.propose(
        SimpleNamespace(settings={"fwhm_min_fit_r2": 0.5}, scorecard=()), lambda: applied.append(1)
    )

    assert message.startswith("Optimised FWHM settings ready to review below: 1 change.")
    assert applied == []
    review.apply()
    assert applied == [1]
    assert panel._haemolynx_report() == "Applied the optimised FWHM settings."


def test_on_clear_drops_a_pending_fwhm_proposal(panel):
    from types import SimpleNamespace

    review = panel._haemolynx_optimise_fwhm_review
    review.propose(SimpleNamespace(settings={"fwhm_min_fit_r2": 0.5}, scorecard=()), lambda: None)

    panel._haemolynx_clear(ask=False)

    assert not review.pending
