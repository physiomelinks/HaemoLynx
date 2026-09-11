"""``_apply_layer_groups`` paces multi-group layer application like a live run.

A live pipeline run never applies more than one stage's layers per Qt event-
loop cycle -- each stage's group crosses a worker-thread signal to the GUI
thread as its own queued event. Loading a saved run or reverting to an
earlier stage both used to build every group inside one plain ``for`` loop
in a single GUI-thread call, with no event-loop cycle in between -- unlike
the live path, which never crashed napari. These tests pin the fix: each
group still gets applied, in order, and the event loop gets a cycle between
every one of them.

Uses `qapp` (pytest-qt) for a real `QApplication` without a real napari
Viewer -- `_apply_layers` itself is mocked out, so a plain sentinel object
stands in for the viewer, same precedent as
`test_gui_optimise_settings_widget.py`.
"""
from __future__ import annotations

import pytest

pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402

pytestmark = pytest.mark.gui


def test_apply_layer_groups_applies_every_group_in_order(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(
        widget_mod, "_apply_layers", lambda viewer, group, report=None: calls.append(group)
    )

    groups = ["segment", "skeletonise", "graph"]
    viewer = object()
    widget_mod._apply_layer_groups(viewer, groups)

    assert calls == groups


def test_apply_layer_groups_pumps_the_qt_event_loop_between_groups(qapp, monkeypatch):
    from qtpy.QtWidgets import QApplication

    monkeypatch.setattr(widget_mod, "_apply_layers", lambda viewer, group, report=None: None)
    process_calls = []
    monkeypatch.setattr(
        QApplication, "processEvents", lambda *a, **k: process_calls.append(1)
    )

    widget_mod._apply_layer_groups(object(), ["a", "b", "c"])

    assert len(process_calls) == 3


def test_apply_layer_groups_forwards_report(qapp, monkeypatch):
    captured = []
    monkeypatch.setattr(
        widget_mod,
        "_apply_layers",
        lambda viewer, group, report=None: captured.append(report),
    )

    sentinel_report = object()
    widget_mod._apply_layer_groups(object(), ["a"], sentinel_report)

    assert captured == [sentinel_report]


def test_apply_layer_groups_handles_zero_groups(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(
        widget_mod, "_apply_layers", lambda viewer, group, report=None: calls.append(group)
    )

    widget_mod._apply_layer_groups(object(), [])

    assert calls == []
