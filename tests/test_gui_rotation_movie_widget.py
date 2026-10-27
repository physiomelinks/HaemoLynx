"""The view window's Rotation movie box, on a real panel and canvas."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.chrome_tooltips import ROTATION_TOOLTIPS  # noqa: E402
from haemolynx.gui.rotation_movie import (  # noqa: E402
    RECORD_LABEL,
    ROTATION_NO_OUTPUT_FOLDER,
    ROTATION_NO_VIEWER,
    ROTATION_NOTHING_SHOWN,
    ROTATION_STEM,
    RotationMovie,
    displayed_axes,
    to_displayed,
    turntable_directions,
    viewer_camera,
)

pytestmark = pytest.mark.gui


def _volume() -> np.ndarray:
    """An L of bright voxels: no two quarter turns of it look alike."""
    volume = np.zeros((12, 16, 20), dtype=np.uint8)
    volume[2:10, 3:5, 3:17] = 255
    volume[2:10, 3:13, 3:5] = 255
    return volume


def _panel_on(viewer, tmp_path):
    panel = settings_widget(napari_viewer=viewer)
    panel._haemolynx_rows()["vtk_output_prefix"].value = tmp_path / "artifacts" / "run"
    return panel, tmp_path / "artifacts"


def _fake_canvas(viewer, monkeypatch, *, shape=(21, 31, 4)):
    """Screenshots that record the camera they were taken at, and carry
    their own index in every pixel."""
    seen = []

    def screenshot(**_kwargs):
        seen.append(
            (tuple(viewer_camera(viewer).view_direction), tuple(viewer_camera(viewer).up_direction))
        )
        return np.full(shape, (len(seen) * 7) % 250, dtype=np.uint8)

    monkeypatch.setattr(viewer.window, "screenshot", screenshot)
    return seen


def _read_movie(path):
    import imageio.v2 as iio

    if path.suffix == ".gif":
        return [np.asarray(frame)[..., :3] for frame in iio.mimread(path)]
    reader = iio.get_reader(path, format="FFMPEG")
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


def _camera(viewer):
    return (
        int(viewer.dims.ndisplay),
        tuple(viewer_camera(viewer).angles),
        tuple(viewer_camera(viewer).center),
        float(viewer_camera(viewer).zoom),
        tuple(viewer.dims.point),
    )


def test_the_box_sits_under_snapshot_with_its_defaults(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    box = panel._haemolynx_rotation
    group = panel._haemolynx_rotation_group
    layout = panel._haemolynx_view_panel.layout()

    assert group.objectName() == "haemolynx_rotation_group"
    assert group.parentWidget() is panel._haemolynx_view_panel
    assert layout.indexOf(group) == layout.indexOf(panel._haemolynx_snapshot_group) + 1
    assert box.tilt.value() == 30.0
    assert box.speed.value() == 30.0
    assert box.tilt.minimum() == -90.0 and box.tilt.maximum() == 90.0
    assert box.summary.text() == "360 frames, 12.0 s at 30 fps"
    assert box.record.text() == RECORD_LABEL
    assert box.tilt.toolTip() == ROTATION_TOOLTIPS["tilt"]
    assert box.speed.toolTip() == ROTATION_TOOLTIPS["speed"]
    assert box.record.toolTip() == ROTATION_TOOLTIPS["record"]

    box.speed.setValue(90.0)
    assert box.summary.text() == "120 frames, 4.0 s at 30 fps"


def test_record_turns_the_camera_once_round_z_and_writes_every_frame(
    make_napari_viewer, tmp_path, monkeypatch, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_image(_volume(), scale=(2.0, 1.0, 0.5))
    viewer.dims.ndisplay = 3
    viewer_camera(viewer).zoom = 1.7
    before = _camera(viewer)
    panel, out_dir = _panel_on(viewer, tmp_path)
    box = panel._haemolynx_rotation
    seen = _fake_canvas(viewer, monkeypatch)
    box.tilt.setValue(20.0)
    box.speed.setValue(360.0)

    box.record.click()
    assert box.recording
    assert not box.tilt.isEnabled() and not box.speed.isEnabled()
    qtbot.waitUntil(lambda: not box.recording, timeout=20000)

    movie = RotationMovie(tilt_deg=20.0, speed_deg_per_s=360.0)
    assert len(seen) == movie.frame_count == 30
    axes = displayed_axes(viewer.dims.displayed, viewer.dims.ndim)
    # napari's default 3D camera is turned 0 round z, so the turn starts there.
    for (view, up), azimuth in zip(seen, movie.azimuths(0.0)):
        want_view, want_up = turntable_directions(azimuth, 20.0)
        np.testing.assert_allclose(view, to_displayed(want_view, axes), atol=1e-6)
        np.testing.assert_allclose(up, to_displayed(want_up, axes), atol=1e-6)

    written = sorted(out_dir.glob(f"{ROTATION_STEM}_*"))
    assert len(written) == 1 and written[0] == box.path
    assert str(written[0]) in panel._haemolynx_report()
    frames = _read_movie(written[0])
    assert len(frames) == 30
    # Even-sized for H.264, alpha dropped, each frame its own screenshot.
    assert frames[0].shape == (20, 30, 3)
    means = [float(frame.mean()) for frame in frames]
    np.testing.assert_allclose(means, [(i * 7) % 250 for i in range(1, 31)], atol=4)

    assert _camera(viewer) == before
    assert box.record.text() == RECORD_LABEL
    assert box.tilt.isEnabled() and box.speed.isEnabled()


def test_the_turn_starts_where_the_camera_already_looks_round_z(
    make_napari_viewer, tmp_path, monkeypatch, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_image(_volume())
    viewer.dims.ndisplay = 3
    view, up = turntable_directions(125.0, -10.0)
    viewer_camera(viewer).set_view_direction(view, up)
    panel, _out_dir = _panel_on(viewer, tmp_path)
    box = panel._haemolynx_rotation
    seen = _fake_canvas(viewer, monkeypatch)
    box.tilt.setValue(-10.0)
    box.speed.setValue(360.0)

    box.record.click()
    qtbot.waitUntil(lambda: not box.recording, timeout=20000)

    np.testing.assert_allclose(seen[0][0], view, atol=1e-6)
    np.testing.assert_allclose(seen[0][1], up, atol=1e-6)


def test_a_2d_view_is_recorded_in_3d_and_put_back(
    make_napari_viewer, tmp_path, monkeypatch, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_image(_volume())
    viewer.dims.set_point(0, 5)
    viewer_camera(viewer).zoom = 2.5
    assert viewer.dims.ndisplay == 2
    before = _camera(viewer)
    panel, out_dir = _panel_on(viewer, tmp_path)
    box = panel._haemolynx_rotation
    displays = []

    def screenshot(**_kwargs):
        displays.append(int(viewer.dims.ndisplay))
        return np.zeros((10, 10, 4), dtype=np.uint8)

    monkeypatch.setattr(viewer.window, "screenshot", screenshot)
    box.speed.setValue(360.0)
    box.record.click()
    qtbot.waitUntil(lambda: not box.recording, timeout=20000)

    assert displays == [3] * 30
    assert len(list(out_dir.glob(f"{ROTATION_STEM}_*"))) == 1
    assert _camera(viewer) == before


def test_reordered_dims_still_turn_about_the_stack_axis(
    make_napari_viewer, tmp_path, monkeypatch, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_image(_volume())
    viewer.dims.order = (1, 0, 2)
    viewer.dims.ndisplay = 3
    panel, _out_dir = _panel_on(viewer, tmp_path)
    box = panel._haemolynx_rotation
    seen = _fake_canvas(viewer, monkeypatch)
    box.tilt.setValue(0.0)
    box.speed.setValue(360.0)

    box.record.click()
    qtbot.waitUntil(lambda: not box.recording, timeout=20000)

    # z is the second displayed dim: up runs along it in every frame.
    for _view, up in seen:
        np.testing.assert_allclose(up, (0.0, -1.0, 0.0), atol=1e-6)


def test_stop_writes_nothing_and_puts_the_view_back(
    make_napari_viewer, tmp_path, monkeypatch, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_image(_volume())
    viewer.dims.ndisplay = 3
    before = _camera(viewer)
    panel, out_dir = _panel_on(viewer, tmp_path)
    box = panel._haemolynx_rotation
    seen = _fake_canvas(viewer, monkeypatch)
    box.speed.setValue(1.0)

    box.record.click()
    qtbot.waitUntil(lambda: len(seen) >= 3, timeout=20000)
    assert box.record.text().startswith("Stop recording (")
    box.record.click()

    assert not box.recording
    assert 3 <= len(seen) < RotationMovie(speed_deg_per_s=1.0).frame_count
    assert box.path is None
    assert not list(out_dir.glob(f"{ROTATION_STEM}_*"))
    assert "no movie written" in panel._haemolynx_report()
    assert _camera(viewer) == before
    assert box.record.text() == RECORD_LABEL
    count = len(seen)
    qtbot.wait(50)
    assert len(seen) == count


def test_a_stop_during_a_capture_ends_cleanly(
    make_napari_viewer, tmp_path, monkeypatch, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_image(_volume())
    viewer.dims.ndisplay = 3
    panel, out_dir = _panel_on(viewer, tmp_path)
    box = panel._haemolynx_rotation
    taken = []

    def screenshot(**_kwargs):
        taken.append(1)
        if len(taken) == 4:
            box.stop()
        return np.zeros((10, 10, 4), dtype=np.uint8)

    monkeypatch.setattr(viewer.window, "screenshot", screenshot)
    box.speed.setValue(360.0)
    box.record.click()
    qtbot.waitUntil(lambda: not box.recording, timeout=20000)
    qtbot.wait(50)

    assert len(taken) == 4
    assert "no movie written" in panel._haemolynx_report()
    assert not list(out_dir.glob(f"{ROTATION_STEM}_*"))


def test_a_canvas_that_cannot_be_captured_leaves_no_file(
    make_napari_viewer, tmp_path, monkeypatch, qtbot
):
    viewer = make_napari_viewer()
    viewer.add_image(_volume())
    viewer.dims.ndisplay = 3
    before = _camera(viewer)
    panel, out_dir = _panel_on(viewer, tmp_path)
    box = panel._haemolynx_rotation
    calls = []

    def screenshot(**_kwargs):
        calls.append(1)
        if len(calls) > 2:
            raise RuntimeError("canvas gone")
        return np.zeros((10, 10, 4), dtype=np.uint8)

    monkeypatch.setattr(viewer.window, "screenshot", screenshot)
    box.speed.setValue(360.0)
    box.record.click()
    qtbot.waitUntil(lambda: not box.recording, timeout=20000)

    assert panel._haemolynx_report().startswith("Could not record the rotation")
    assert not list(out_dir.glob(f"{ROTATION_STEM}_*"))
    assert _camera(viewer) == before


def test_without_an_output_folder_or_anything_shown_nothing_records(
    make_napari_viewer, tmp_path, monkeypatch
):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    box = panel._haemolynx_rotation
    seen = _fake_canvas(viewer, monkeypatch)

    panel._haemolynx_rows()["vtk_output_prefix"].value = ""
    viewer.add_image(_volume())
    box.record.click()
    assert panel._haemolynx_report() == ROTATION_NO_OUTPUT_FOLDER
    assert not box.recording

    panel._haemolynx_rows()["vtk_output_prefix"].value = tmp_path / "run"
    viewer.layers[0].visible = False
    box.record.click()
    assert panel._haemolynx_report() == ROTATION_NOTHING_SHOWN
    assert not box.recording
    assert seen == []


def test_record_with_no_viewer_says_so(qapp):
    # qapp: with no viewer, nothing else makes the QApplication the panel needs.
    panel = settings_widget(napari_viewer=None)
    panel._haemolynx_rotation.record.click()
    assert panel._haemolynx_report() == ROTATION_NO_VIEWER
    assert not panel._haemolynx_rotation.recording


def test_a_real_canvas_turns_in_the_movie(make_napari_viewer, tmp_path, qtbot):
    """No stand-in screenshots: the canvas renders each frame, and a quarter
    turn of the L looks different from where it started. Shown, since a
    hidden canvas's screenshot does not follow the camera."""
    viewer = make_napari_viewer(show=True)
    viewer.add_image(_volume(), rendering="mip")
    viewer.dims.ndisplay = 3
    panel, out_dir = _panel_on(viewer, tmp_path)
    box = panel._haemolynx_rotation
    box.speed.setValue(360.0)

    box.record.click()
    qtbot.waitUntil(lambda: not box.recording, timeout=60000)

    assert box.path is not None, panel._haemolynx_report()
    frames = _read_movie(box.path)
    assert len(frames) == 30
    assert all(frame.shape == frames[0].shape for frame in frames)
    quarter = np.abs(frames[0].astype(float) - frames[7].astype(float)).mean()
    assert quarter > 1.0
    assert any(frame.max() > 100 for frame in frames)
