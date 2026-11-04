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


def test_optimise_fwhm_button_sits_right_after_the_measure_row(panel):
    """Per the feature request: 'under the master checkbox and input file
    line for FWHM, but before the other settings'. The raw image is on the
    Input tab now, so the master checkbox is the line above it."""
    container = panel._haemolynx_diameters_settings
    widgets = list(container)
    rows = panel._haemolynx_rows()
    measure_index = widgets.index(rows["measure_diameters_from_raw_image"])
    button_index = widgets.index(panel._haemolynx_optimise_fwhm_button)
    assert button_index == measure_index + 1
    # And before the next FWHM setting row.
    next_row_index = widgets.index(rows["fwhm_sample_spacing_along_edge_um"])
    assert button_index < next_row_index


def test_optimise_fwhm_button_hidden_until_fwhm_measurement_is_turned_on(panel, tmp_path):
    """A nested widget's own ``.visible`` only reflects reality once its
    top-level ancestor is shown *and* its tab is the active one (an inactive
    QTabWidget page stays hidden regardless) -- see
    test_gui_optimise_settings_widget.py's own
    test_ticking_choose_groups_reveals_the_checkbox_list for the same
    caveat; that one only needed `.show()` because its widget lives on the
    already-active "1. Input" tab, while this button lives on "5. Diameters".
    FWHM can run only with a raw image to read, so one is set first."""
    from haemolynx.gui.tabs import tab_titles

    panel.show()
    diameters_index = list(tab_titles()).index("5. Diameters")
    panel._haemolynx_tabs.setCurrentIndex(diameters_index)
    rows = panel._haemolynx_rows()
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    rows["raw_image_path"].value = raw_file
    rows["measure_diameters_from_raw_image"].value = False
    assert panel._haemolynx_optimise_fwhm_button.visible is False

    rows["measure_diameters_from_raw_image"].value = True
    assert panel._haemolynx_optimise_fwhm_button.visible is True

    rows["measure_diameters_from_raw_image"].value = False
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
    assert "Choose the raw image on the Input tab" in panel._haemolynx_report()


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
    panel._haemolynx_rows()["raw_image_path"].value = raw_file

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
    panel._haemolynx_rows()["raw_image_path"].value = raw_file

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
    checkpoint (a loaded one) hands none, and the worker reads it from disk."""
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
    panel._haemolynx_rows()["raw_image_path"].value = raw_file

    panel._haemolynx_optimise_fwhm_settings()

    assert (started[0]["vessel_mask"] is mask) if skeletonised else started[0]["vessel_mask"] is None


@pytest.mark.parametrize("has_boundaries", [True, False])
def test_optimise_fwhm_settings_hands_the_worker_the_runs_boundaries(
    panel, monkeypatch, tmp_path, has_boundaries
):
    """The boundary nodes decide which vessels boundary_handling keeps, and
    so which a run calibrates its widths on; a run without a Boundaries
    checkpoint (stopped at Graph, or loaded) calibrates on every vessel."""
    from haemolynx.gui.stage_checkpoints import StageCheckpoint

    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *args, **kwargs: started.append(kwargs),
    )
    if has_boundaries:
        panel._haemolynx_checkpoints.replace_all(
            [StageCheckpoint(stage="assign_boundaries", title="4. Boundaries", group=None,
                             inlet_nodes=(0,), outlet_nodes=(1,))]
        )
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    panel._haemolynx_rows()["raw_image_path"].value = raw_file

    panel._haemolynx_optimise_fwhm_settings()

    expected = panel._haemolynx_checkpoints.get("assign_boundaries") if has_boundaries else None
    assert started[0]["boundaries"] is expected


def _branched_graph():
    """0 - 1 - 2 with a dead end 1 - 3 off it: with inlet 0 and outlet 2,
    remove_disconnected removes the dead end."""
    import networkx as nx
    import numpy as np

    G = nx.MultiGraph()
    ends = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 10.0), 2: (0.0, 0.0, 20.0), 3: (0.0, 10.0, 10.0)}
    for node, pos in ends.items():
        G.add_node(node, pos=np.asarray(pos))
    for u, v in ((0, 1), (1, 2), (1, 3)):
        a, b = np.asarray(ends[u]), np.asarray(ends[v])
        G.add_edge(u, v, length=10.0, voxels=[tuple(a + (b - a) * t / 10.0) for t in range(11)])
    return G


@pytest.mark.parametrize("handling", ["leave_unsolved", "remove_disconnected"])
def test_the_fwhm_optimiser_measures_with_the_runs_config_and_calibration_edges(
    monkeypatch, tmp_path, handling
):
    """Regression: the optimiser built its own FWHM call and calibrated on
    every vessel. It now gets the run's haemodynamics config, as the
    Diameters stage builds it, and the vessels network handling keeps."""
    from haemolynx.pipeline import default_schema, resolve_settings, stages

    calls = []
    monkeypatch.setattr(
        widget_mod, "optimise_fwhm_settings", lambda graph, **kwargs: calls.append(kwargs) or "done"
    )
    monkeypatch.setattr(widget_mod, "_run_segmentation_on_its_grid", lambda _settings: None)
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    schema = default_schema()
    settings = resolve_settings(
        {
            "input_path": raw_file,
            "raw_image_path": raw_file,
            "measure_diameters_from_raw_image": True,
            "boundary_handling": handling,
        },
        schema=schema,
        config_path=None,
    )
    graph = _branched_graph()
    boundaries = SimpleNamespace(inlet_nodes=(0,), outlet_nodes=(2,))

    result = widget_mod._optimise_fwhm_settings_for_run(
        graph, (2.0, 1.0, 1.0), settings, schema, boundaries=boundaries, progress=None,
        groups=None, vessel_mask=None, sample_edge_count=None, max_passes=1,
    )

    assert result == "done"
    kwargs = calls[0]
    assert kwargs["config"] == stages._haemodynamics_apply_config(
        settings, schema, voxel_size_zyx=(2.0, 1.0, 1.0),
        stain=stages.recorded_raw_image_stain(settings, graph),
    )
    assert kwargs["calibration_edges"] == (
        None if handling == "leave_unsolved" else [(0, 1, 0), (1, 2, 0)]
    )


def test_the_settings_the_fwhm_optimiser_reads_are_real_settings():
    from haemolynx.pipeline.schema import SCHEMA

    assert set(widget_mod._FWHM_OPTIMISER_READS) <= {setting.name for setting in SCHEMA}


@pytest.mark.parametrize("optimiser", ["optimise_fwhm_settings", "optimise_endothelial_settings"])
def test_after_load_run_the_diameter_optimisers_read_the_runs_segmentation(
    monkeypatch, tmp_path, optimiser
):
    """Regression: a loaded run keeps no segmented image in memory, so both
    optimisers ran their trials without one -- no line stopped at the vessel
    beside, and no decoys or planted vessels -- where a run resumed there
    reads it from disk. They now read it as the run's skeletonise does: cut
    and reduced onto its grid, binarised."""
    import numpy as np
    import tifffile

    from haemolynx.pipeline import default_schema, resolve_settings

    calls = []
    monkeypatch.setattr(widget_mod, optimiser, lambda graph, **kwargs: calls.append(kwargs) or "done")
    segmented = np.zeros((4, 6, 8), dtype=np.uint8)
    segmented[1:3, 2:4, 1:7] = 255
    segmented_file = tmp_path / "segmented.tif"
    tifffile.imwrite(segmented_file, segmented)
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    schema = default_schema()
    endothelial = optimiser == "optimise_endothelial_settings"
    settings = resolve_settings(
        {
            "input_path": segmented_file,
            "raw_image_path": raw_file,
            "raw_image_stain": "endothelial" if endothelial else "plasma_column",
            "measure_diameters_from_raw_image": True,
        },
        schema=schema,
        config_path=None,
    )
    run_for = (
        widget_mod._optimise_endothelial_settings_for_run if endothelial
        else widget_mod._optimise_fwhm_settings_for_run
    )

    run_for(
        _branched_graph(), (1.0, 1.0, 1.0), settings, schema, boundaries=None, progress=None,
        groups=None, vessel_mask=None, sample_edge_count=None, max_passes=1,
    )

    np.testing.assert_array_equal(calls[0]["vessel_mask"], segmented > 0)


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
    panel._haemolynx_rows()["raw_image_path"].value = raw_file
    panel._haemolynx_rows()["measure_diameters_from_raw_image"].value = True

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
    panel._haemolynx_rows()["raw_image_path"].value = raw_file
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


def test_the_fwhm_options_and_review_show_only_with_fwhm_measurement(panel, tmp_path):
    from haemolynx.gui.tabs import tab_titles

    panel.show()
    panel._haemolynx_tabs.setCurrentIndex(list(tab_titles()).index("5. Diameters"))
    rows = panel._haemolynx_rows()
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    rows["raw_image_path"].value = raw_file
    box = panel._haemolynx_optimise_fwhm_box

    rows["measure_diameters_from_raw_image"].value = False
    assert box.visible is False
    rows["measure_diameters_from_raw_image"].value = True
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
        SimpleNamespace(settings={"fwhm_min_fit_r2": 0.5}, scorecard=()), applied.append
    )

    assert message.startswith("Optimised FWHM settings ready to review below: 1 change.")
    assert applied == []
    review.apply()
    assert applied == [("fwhm_min_fit_r2",)]
    assert panel._haemolynx_report() == "Applied the optimised FWHM settings."


def test_apply_writes_only_the_settings_the_review_listed(panel):
    """Regression: Apply wrote every setting the optimiser returned, so a
    row the review said would not change, edited since, was put back."""
    from types import SimpleNamespace

    review = panel._haemolynx_optimise_fwhm_review
    model = panel._haemolynx_rows()["fwhm_profile_model"].value
    written = []
    review.propose(
        SimpleNamespace(
            settings={"fwhm_min_fit_r2": 0.5, "fwhm_profile_model": model}, scorecard=()
        ),
        written.append,
    )

    review.apply()

    assert written == [("fwhm_min_fit_r2",)]


def test_apply_writes_the_listed_settings_and_leaves_a_row_edited_since(panel, monkeypatch, tmp_path):
    """The run's own writer, as the finished optimisation hands it to the
    review: Apply writes what the review listed and nothing else, so a row
    edited after the review keeps the edit."""
    from haemolynx.optimisation.search import OptimisationResult

    def finish_at_once(_run, *, run_state, cancel_flag, finished, **_kwargs):
        run_state.start(worker=None, cancel_flag=cancel_flag)
        finished((result, raw_file))

    monkeypatch.setattr(widget_mod, "_start_optimisation_worker", finish_at_once)
    rows = panel._haemolynx_rows()
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    rows["raw_image_path"].value = raw_file
    rows["measure_diameters_from_raw_image"].value = True
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    unchanged = rows["fwhm_min_accepted_samples"].value
    result = OptimisationResult(
        settings={"fwhm_min_fit_r2": 0.5, "fwhm_min_accepted_samples": unchanged}, trials=()
    )

    panel._haemolynx_optimise_fwhm_settings()
    review = panel._haemolynx_optimise_fwhm_review
    assert review.pending
    rows["fwhm_min_accepted_samples"].value = unchanged + 3  # edited after the review
    review.apply()

    assert rows["fwhm_min_fit_r2"].value == pytest.approx(0.5)
    assert rows["fwhm_min_accepted_samples"].value == unchanged + 3


def test_loading_a_config_drops_a_pending_fwhm_proposal(panel, tmp_path):
    """Its changes were listed against the settings the config replaces."""
    from types import SimpleNamespace

    review = panel._haemolynx_optimise_fwhm_review
    review.propose(SimpleNamespace(settings={"fwhm_min_fit_r2": 0.5}, scorecard=()), lambda _names: None)
    config = tmp_path / "config.yaml"
    config.write_text("fwhm_min_fit_r2: 0.9\n", encoding="utf-8")

    panel._haemolynx_load_config(config)

    assert not review.pending


def test_on_clear_drops_a_pending_fwhm_proposal(panel):
    from types import SimpleNamespace

    review = panel._haemolynx_optimise_fwhm_review
    review.propose(SimpleNamespace(settings={"fwhm_min_fit_r2": 0.5}, scorecard=()), lambda _names: None)

    panel._haemolynx_clear(ask=False)

    assert not review.pending


# --- a synthetic network: no image to optimise on (PLAN_optimisers.md, Phase 5)


def test_optimise_fwhm_settings_greys_and_refuses_for_a_synthetic_network(panel, monkeypatch, tmp_path):
    from haemolynx.gui.chrome_tooltips import OPTIMISE_SYNTHETIC_TOOLTIP

    started = []
    monkeypatch.setattr(
        widget_mod, "_run_fwhm_optimisation_in_background", lambda *a, **k: started.append((a, k))
    )
    button = panel._haemolynx_optimise_fwhm_button
    before = (button.enabled, button.tooltip)
    rows = panel._haemolynx_rows()
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    rows["raw_image_path"].value = raw_file

    rows["network_source"].value = "synthetic"
    assert not button.enabled
    assert button.tooltip == OPTIMISE_SYNTHETIC_TOOLTIP
    panel._haemolynx_optimise_fwhm_settings()
    assert not started
    assert panel._haemolynx_report() == OPTIMISE_SYNTHETIC_TOOLTIP

    rows["network_source"].value = "segmented_image"
    assert (button.enabled, button.tooltip) == before
