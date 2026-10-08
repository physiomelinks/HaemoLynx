"""visualize_overlay must survive a failing depth-peeling call (headless render)."""
import sys

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from ImageLynx.visualization import plot


class _FakePlotter:
    def __init__(self, *args, **kwargs):
        pass

    def set_background(self, *args, **kwargs):
        pass

    def enable_depth_peeling(self, *args, **kwargs):
        raise RuntimeError("no GL context")

    def add_mesh(self, *args, **kwargs):
        pass

    def add_points(self, *args, **kwargs):
        pass

    def add_axes(self):
        pass

    def add_legend(self):
        pass


def test_failed_depth_peeling_is_logged_not_a_name_error(monkeypatch, caplog):
    monkeypatch.setattr(pv, "Plotter", _FakePlotter)
    # The depth-peeling branch is skipped while pytest is imported.
    monkeypatch.delitem(sys.modules, "pytest")
    volume = np.zeros((4, 4, 4), dtype=bool)
    volume[1:3, 1:3, 1:3] = True

    with caplog.at_level("WARNING"):
        plot.visualize_overlay(volume, volume.copy(), show=False)

    assert "Could not enable depth peeling" in caplog.text
