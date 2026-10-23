"""The "Optimise endothelial settings" button, built for real, with a live
QApplication -- the counterpart of tests/test_gui_fwhm_optimise_widget.py for
a raw image that stains the vessel wall, where the ring fit reads diameters
instead of FWHM.
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


def _endothelial_run(panel, tmp_path, *, stain="endothelial"):
    """The panel set up for a run whose raw image stains the wall, with a
    network to optimise on."""
    rows = panel._haemolynx_rows()
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    rows["raw_image_path"].value = raw_file
    rows["raw_image_stain"].value = stain
    rows["measure_diameters_from_raw_image"].value = True
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    return rows


def _on_the_diameters_tab(panel):
    """A nested widget's ``visible`` reflects reality only once the panel is
    shown and its tab is the current one (see the FWHM button's tests)."""
    from haemolynx.gui.tabs import tab_titles

    panel.show()
    panel._haemolynx_tabs.setCurrentIndex(list(tab_titles()).index("5. Diameters"))


def _capture_starts(monkeypatch):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_endothelial_optimisation_in_background",
        lambda *args, **kwargs: started.append((args, kwargs)),
    )
    return started


# --- the button exists, where and when ---------------------------------------


def test_the_button_exists_with_hover_text(panel):
    button = panel._haemolynx_optimise_endothelial_button
    assert button.text == "Optimise endothelial settings"
    assert button.tooltip.strip()
    for checkbox in panel._haemolynx_optimise_endothelial_group_checkboxes.values():
        assert checkbox.tooltip.strip()
    for widget in (
        panel._haemolynx_optimise_endothelial_options,
        panel._haemolynx_optimise_endothelial_choose_groups,
        panel._haemolynx_optimise_endothelial_sample,
        panel._haemolynx_optimise_endothelial_passes,
    ):
        assert widget.tooltip.strip()


def test_it_sits_where_the_fwhm_button_does_with_its_box_under_it(panel):
    widgets = list(panel._haemolynx_diameters_settings)
    fwhm_box = widgets.index(panel._haemolynx_optimise_fwhm_box)
    button = widgets.index(panel._haemolynx_optimise_endothelial_button)
    assert button == fwhm_box + 1
    assert widgets.index(panel._haemolynx_optimise_endothelial_box) == button + 1
    rows = panel._haemolynx_rows()
    assert button < widgets.index(rows["fwhm_sample_spacing_along_edge_um"])


def test_it_is_shown_exactly_when_the_ring_fit_would_run(panel, tmp_path):
    """The ring fit runs when diameters are read from a raw image that
    stains the wall: then this button shows and the FWHM one does not, and
    the other way round for a plasma label."""
    _on_the_diameters_tab(panel)
    rows = _endothelial_run(panel, tmp_path)
    endothelial = panel._haemolynx_optimise_endothelial_button
    fwhm = panel._haemolynx_optimise_fwhm_button

    assert endothelial.visible is True and fwhm.visible is False
    assert panel._haemolynx_optimise_endothelial_box.visible is True

    rows["raw_image_stain"].value = "plasma_column"
    assert endothelial.visible is False and fwhm.visible is True

    rows["raw_image_stain"].value = "endothelial"
    rows["measure_diameters_from_raw_image"].value = False
    assert endothelial.visible is False and fwhm.visible is False
    assert panel._haemolynx_optimise_endothelial_box.visible is False


def test_an_auto_stain_shows_it_only_once_a_run_read_it_as_endothelial(panel, tmp_path):
    _on_the_diameters_tab(panel)
    _endothelial_run(panel, tmp_path, stain="auto")
    assert panel._haemolynx_optimise_endothelial_button.visible is False


def test_it_is_hidden_and_refuses_for_a_synthetic_network(panel, tmp_path, monkeypatch):
    started = _capture_starts(monkeypatch)
    _on_the_diameters_tab(panel)
    rows = _endothelial_run(panel, tmp_path)

    rows["network_source"].value = "synthetic"

    assert panel._haemolynx_optimise_endothelial_button.visible is False
    panel._haemolynx_optimise_endothelial_settings()
    assert not started
    assert "synthetic network reads no image" in panel._haemolynx_report()


# --- guards ------------------------------------------------------------------


def test_it_refuses_without_a_graph(panel, monkeypatch):
    started = _capture_starts(monkeypatch)
    panel._haemolynx_optimise_endothelial_settings()
    assert not started
    assert "run the pipeline through at least Graph first" in panel._haemolynx_report()


def test_it_refuses_without_a_raw_image(panel, monkeypatch):
    started = _capture_starts(monkeypatch)
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    panel._haemolynx_optimise_endothelial_settings()
    assert not started
    assert "Choose the raw image on the Input tab" in panel._haemolynx_report()


def test_it_refuses_while_a_run_is_going(panel, monkeypatch, tmp_path):
    started = _capture_starts(monkeypatch)
    _endothelial_run(panel, tmp_path)
    panel._haemolynx_run_state.start(worker=None, cancel_flag={"cancelled": False})
    try:
        panel._haemolynx_optimise_endothelial_settings()
    finally:
        panel._haemolynx_run_state.stopped()
    assert not started
    assert "already going" in panel._haemolynx_report()


# --- wiring -------------------------------------------------------------------


def test_it_starts_the_worker_with_the_runs_state(panel, monkeypatch, tmp_path):
    import numpy as np

    from haemolynx.gui.stage_checkpoints import StageCheckpoint

    started = _capture_starts(monkeypatch)
    _endothelial_run(panel, tmp_path)
    graph = _tiny_graph()
    panel._haemolynx_view.results = _fake_results(graph, voxel_size_zyx=(2.0, 1.0, 1.0))
    mask = np.ones((3, 3, 11), dtype=bool)
    panel._haemolynx_checkpoints.replace_all([
        StageCheckpoint(stage="skeletonise", title="2. Skeletonise", group=None,
                        output=SimpleNamespace(image=mask)),
        StageCheckpoint(stage="assign_boundaries", title="4. Boundaries", group=None,
                        inlet_nodes=(0,), outlet_nodes=(1,)),
    ])

    panel._haemolynx_optimise_endothelial_settings()

    (args, kwargs), = started
    passed_graph, voxel_size_zyx, _settings, _schema, rows, _report, button, bars = args
    assert passed_graph is not graph and passed_graph.number_of_edges() == 1
    assert voxel_size_zyx == (2.0, 1.0, 1.0)
    assert rows is panel._haemolynx_rows()
    assert button is panel._haemolynx_optimise_endothelial_button
    assert bars is panel._haemolynx_optimise_endothelial_bars
    assert kwargs["vessel_mask"] is mask
    assert kwargs["boundaries"] is panel._haemolynx_checkpoints.get("assign_boundaries")
    assert kwargs["run_state"] is panel._haemolynx_run_state
    assert kwargs["groups"] is None
    assert kwargs["sample_edge_count"] is None  # Auto
    assert kwargs["max_passes"] == 1
    assert kwargs["propose"] == panel._haemolynx_optimise_endothelial_review.propose


def test_the_options_are_passed_through(panel, monkeypatch, tmp_path):
    started = _capture_starts(monkeypatch)
    _endothelial_run(panel, tmp_path)
    panel._haemolynx_optimise_endothelial_choose_groups.value = True
    for name, box in panel._haemolynx_optimise_endothelial_group_checkboxes.items():
        box.value = name == "ring_gate"
    panel._haemolynx_optimise_endothelial_sample.value = "25 vessels"
    panel._haemolynx_optimise_endothelial_passes.value = 3

    panel._haemolynx_optimise_endothelial_settings()

    kwargs = started[0][1]
    assert kwargs["groups"] == ("ring_gate",)
    assert kwargs["sample_edge_count"] == 25
    assert kwargs["max_passes"] == 3


def test_the_options_open_under_their_toggle(panel, tmp_path):
    _on_the_diameters_tab(panel)
    _endothelial_run(panel, tmp_path)
    container = panel._haemolynx_optimise_endothelial_options_container
    assert container.visible is False
    panel._haemolynx_optimise_endothelial_options.value = True
    assert container.visible is True


def test_clicking_the_native_button_starts_it(panel, monkeypatch, tmp_path):
    started = _capture_starts(monkeypatch)
    _endothelial_run(panel, tmp_path)
    panel._haemolynx_optimise_endothelial_button.native.click()
    assert len(started) == 1


def _branched_graph():
    """0 - 1 - 2 with a dead end 1 - 3: with inlet 0 and outlet 2,
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
def test_it_reads_with_the_runs_config_and_calibration_vessels(monkeypatch, tmp_path, handling):
    """The optimiser gets the haemodynamics config the Diameters stage
    builds -- the ring fit on, the run's raw image and channel -- the
    vessels network handling keeps, and the four settings it decides."""
    from haemolynx.optimisation.endothelial_search import ENDOTHELIAL_SETTING_NAMES
    from haemolynx.pipeline import default_schema, resolve_settings, stages

    calls = []
    monkeypatch.setattr(
        widget_mod, "optimise_endothelial_settings",
        lambda graph, **kwargs: calls.append(kwargs) or "done",
    )
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    schema = default_schema()
    settings = resolve_settings(
        {
            "input_path": raw_file,
            "raw_image_path": raw_file,
            "raw_image_channel": 1,
            "raw_image_stain": "endothelial",
            "measure_diameters_from_raw_image": True,
            "boundary_handling": handling,
            "endothelial_min_ring_contrast": 3.0,
        },
        schema=schema,
        config_path=None,
    )
    graph = _branched_graph()

    result = widget_mod._optimise_endothelial_settings_for_run(
        graph, (2.0, 1.0, 1.0), settings, schema,
        boundaries=SimpleNamespace(inlet_nodes=(0,), outlet_nodes=(2,)), progress=None,
        groups=None, vessel_mask=None, sample_edge_count=None, max_passes=1,
    )

    assert result == "done"
    kwargs = calls[0]
    assert kwargs["config"] == stages._haemodynamics_apply_config(
        settings, schema, voxel_size_zyx=(2.0, 1.0, 1.0), stain="endothelial",
    )
    assert kwargs["config"].endothelial["use_endothelial_diameters"] is True
    assert kwargs["config"].endothelial["endothelial_channel"] == 1
    assert kwargs["calibration_edges"] == (
        None if handling == "leave_unsolved" else [(0, 1, 0), (1, 2, 0)]
    )
    assert set(kwargs["starting_values"]) == set(ENDOTHELIAL_SETTING_NAMES)
    assert kwargs["starting_values"]["endothelial_min_ring_contrast"] == 3.0


def test_the_settings_it_decides_are_real_settings():
    from haemolynx.optimisation.endothelial_search import ENDOTHELIAL_SETTING_NAMES
    from haemolynx.pipeline.schema import SCHEMA

    assert set(ENDOTHELIAL_SETTING_NAMES) <= {setting.name for setting in SCHEMA}


# --- the proposal, and clearing -------------------------------------------------


def test_a_proposal_applies_only_on_apply_and_writes_unset_as_unset(panel):
    review = panel._haemolynx_optimise_endothelial_review
    rows = panel._haemolynx_rows()
    rows["endothelial_wall_thickness_um"].value = 1.5
    applied = []
    proposal = SimpleNamespace(
        settings={"endothelial_wall_thickness_um": None, "endothelial_min_ring_contrast": 3.0},
        scorecard=(),
    )

    message = review.propose(proposal, lambda: applied.append(1))

    assert message.startswith("Optimised endothelial settings ready to review below: 2 changes.")
    assert "endothelial_wall_thickness_um: 1.5 -> unset" in review.text.value
    assert applied == []
    review.apply()
    assert applied == [1]
    assert panel._haemolynx_report() == "Applied the optimised endothelial settings."


def test_clear_resets_the_button_and_drops_a_pending_proposal(panel):
    review = panel._haemolynx_optimise_endothelial_review
    review.propose(
        SimpleNamespace(settings={"endothelial_min_ring_contrast": 3.0}, scorecard=()), lambda: None
    )
    panel._haemolynx_optimise_endothelial_button.enabled = False
    panel._haemolynx_run_state.start(worker=None, cancel_flag={"cancelled": False})

    panel._haemolynx_clear(ask=False)

    assert panel._haemolynx_optimise_endothelial_button.enabled is True
    assert not review.pending
