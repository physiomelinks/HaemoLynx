"""``_apply_layer_groups`` draws what several stage groups leave, once.

Loading a saved run and "Run from this stage" replay every stage up to some
point. Applied one group at a time -- even a Qt event-loop cycle apart, the
earlier fix -- the vessels and nodes were redrawn once per stage, and where a
stage's layer was smaller than the one before it, the layer was removed and
re-added: a GL teardown that crashed napari with an access violation in the
driver on a real 3,191-vessel network. These tests pin the replacement: one
application, of each layer's final state (see
``haemolynx.gui.results.merge_stage_layers``).

Uses `qapp` (pytest-qt) for a real `QApplication` without a real napari
Viewer -- `_apply_layers` itself is mocked out, so a plain sentinel object
stands in for the viewer, same precedent as
`test_gui_optimise_settings_widget.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402
from haemolynx.gui.results import LayerSpec, StageLayers  # noqa: E402

pytestmark = pytest.mark.gui


def _group(stage: str, *names_and_rows: tuple[str, int]) -> StageLayers:
    return StageLayers(
        stage=stage,
        title=stage,
        layers=tuple(
            LayerSpec(kind="points", name=name, data=np.zeros((rows, 3)))
            for name, rows in names_and_rows
        ),
    )


def test_apply_layer_groups_applies_each_layers_final_state_once(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(
        widget_mod, "_apply_layers", lambda viewer, group, report=None: calls.append(group)
    )

    groups = [
        _group("build_network", ("vessels", 9), ("nodes", 5)),
        _group("assign_boundaries", ("vessels", 7)),
        _group("solve", ("flow", 3)),
    ]
    widget_mod._apply_layer_groups(object(), groups)

    assert len(calls) == 1
    applied = calls[0]
    assert [(spec.name, len(spec.data)) for spec in applied.layers] == [
        ("vessels", 7), ("nodes", 5), ("flow", 3),
    ]
    assert applied.stage == "solve"


def test_apply_layer_groups_forwards_report(qapp, monkeypatch):
    captured = []
    monkeypatch.setattr(
        widget_mod,
        "_apply_layers",
        lambda viewer, group, report=None: captured.append(report),
    )

    sentinel_report = object()
    widget_mod._apply_layer_groups(object(), [_group("a", ("x", 1))], sentinel_report)

    assert captured == [sentinel_report]


def test_apply_layer_groups_handles_zero_groups(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(
        widget_mod, "_apply_layers", lambda viewer, group, report=None: calls.append(group)
    )

    widget_mod._apply_layer_groups(object(), [])

    assert calls == []
