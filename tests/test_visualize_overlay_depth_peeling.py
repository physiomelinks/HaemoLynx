"""visualize_overlay must survive a failing depth-peeling call (headless render)."""
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
    volume = np.zeros((4, 4, 4), dtype=bool)
    volume[1:3, 1:3, 1:3] = True

    with caplog.at_level("WARNING"):
        plot.visualize_overlay(volume, volume.copy(), show=False, depth_peeling=True)

    assert "Could not enable depth peeling" in caplog.text


def test_depth_peeling_off_never_asks_the_plotter(monkeypatch, caplog):
    monkeypatch.setattr(pv, "Plotter", _FakePlotter)
    volume = np.zeros((4, 4, 4), dtype=bool)
    volume[1:3, 1:3, 1:3] = True

    with caplog.at_level("WARNING"):
        plot.visualize_overlay(volume, volume.copy(), show=False, depth_peeling=False)

    assert "Could not enable depth peeling" not in caplog.text


def test_depth_peeling_is_on_by_default_whatever_is_imported(monkeypatch, caplog):
    # pytest is imported here; the choice must not depend on that.
    monkeypatch.setattr(pv, "Plotter", _FakePlotter)
    volume = np.zeros((4, 4, 4), dtype=bool)
    volume[1:3, 1:3, 1:3] = True

    with caplog.at_level("WARNING"):
        plot.visualize_overlay(volume, volume.copy(), show=False)

    assert "Could not enable depth peeling" in caplog.text
