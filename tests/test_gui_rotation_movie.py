"""The rotation movie's camera directions, frame counts, frames and file."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from haemolynx.gui.rotation_movie import (
    ROTATION_STEM,
    RotationMovie,
    describe,
    displayed_axes,
    from_displayed,
    movie_frame,
    movie_shape,
    open_movie_writer,
    start_azimuth,
    to_displayed,
    turntable_directions,
    unique_movie_path,
)
from haemolynx.gui.view_snap import camera_directions_for_plane

Z = np.array([1.0, 0.0, 0.0])


def _right(view, up):
    """napari's screen right, ``view x up`` (see ``view_snap``)."""
    return np.cross(np.asarray(view), np.asarray(up))


@pytest.mark.parametrize(
    ("azimuth", "tilt", "plane"),
    [(0.0, 0.0, "XZ"), (90.0, 0.0, "YZ"), (0.0, -90.0, "XY")],
)
def test_the_turn_passes_through_the_snap_views(azimuth, tilt, plane):
    view, up = turntable_directions(azimuth, tilt)
    snap_view, snap_up = camera_directions_for_plane(plane)
    np.testing.assert_allclose(view, snap_view, atol=1e-12)
    np.testing.assert_allclose(up, snap_up, atol=1e-12)


@pytest.mark.parametrize("tilt", [-90.0, -45.0, 0.0, 20.0, 30.0, 89.0, 90.0])
@pytest.mark.parametrize("azimuth", [0.0, 37.0, 90.0, 180.0, 271.5])
def test_every_frame_looks_at_the_tilt_with_z_upright(azimuth, tilt):
    view, up = (np.asarray(v) for v in turntable_directions(azimuth, tilt))
    assert np.linalg.norm(view) == pytest.approx(1.0)
    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert float(view @ up) == pytest.approx(0.0, abs=1e-12)
    # The view leans onto z by the tilt: from above (towards +z, the top
    # slice nearest the camera) for a positive one.
    assert float(view @ Z) == pytest.approx(np.sin(np.radians(tilt)), abs=1e-12)
    # z never leans sideways on screen, and runs down it (top slice at top).
    assert float(_right(view, up) @ Z) == pytest.approx(0.0, abs=1e-12)
    assert float(up @ Z) == pytest.approx(-np.cos(np.radians(tilt)), abs=1e-12)


def test_the_turn_goes_round_z_one_way_at_a_steady_rate():
    movie = RotationMovie(tilt_deg=30.0, speed_deg_per_s=45.0, fps=8)
    views = np.array([turntable_directions(a, 30.0)[0] for a in movie.azimuths()])
    across = views[:, 1:] / np.linalg.norm(views[:, 1:], axis=1, keepdims=True)
    steps = np.degrees(
        np.arctan2(
            across[:-1, 0] * across[1:, 1] - across[:-1, 1] * across[1:, 0],
            np.sum(across[:-1] * across[1:], axis=1),
        )
    )
    np.testing.assert_allclose(steps, steps[0])
    assert abs(steps[0]) == pytest.approx(360.0 / movie.frame_count)
    # z height of the view never changes: a turntable, not a wobble.
    np.testing.assert_allclose(views[:, 0], np.sin(np.radians(30.0)))


@pytest.mark.parametrize("tilt", [-90.0, -60.0, 0.0, 30.0, 90.0])
@pytest.mark.parametrize("azimuth", [0.0, 15.0, 100.0, 200.0, 359.0])
def test_start_azimuth_reads_back_the_turn_of_any_frame(azimuth, tilt):
    view, up = turntable_directions(azimuth, tilt)
    assert start_azimuth(view, up) == pytest.approx(azimuth % 360.0, abs=1e-9)


def test_start_azimuth_of_naparis_default_view_is_zero():
    """napari's 3D camera starts looking down z with y up the screen: the
    turn starts from the XY-like frame, not half a turn round."""
    view, up = camera_directions_for_plane("XY")
    assert start_azimuth(view, up) == pytest.approx(0.0)


def test_start_azimuth_of_a_camera_with_no_turn_is_zero():
    assert start_azimuth((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == 0.0


def test_reordered_dims_still_turn_about_z():
    """A 3D view whose dims were reordered (a 2D snap first) shows y as its
    first displayed axis; the vector handed to napari is permuted to match."""
    axes = displayed_axes((1, 0, 2), 3)
    assert axes == (1, 0, 2)
    view, up = turntable_directions(0.0, 0.0)
    shown = to_displayed(view, axes)
    assert shown == (1.0, 0.0, 0.0)
    assert from_displayed(shown, axes) == view
    # A leading non-spatial axis is skipped; anything else falls back.
    assert displayed_axes((1, 2, 3), 4) == (0, 1, 2)
    assert displayed_axes((2, 1, 3), 4) == (1, 0, 2)
    assert displayed_axes((0, 1, 2), 4) == (0, 1, 2)


def test_frame_count_is_one_full_turn_at_the_speed():
    movie = RotationMovie(tilt_deg=10.0, speed_deg_per_s=30.0)
    assert movie.fps == 30
    assert movie.frame_count == 360
    assert movie.seconds == pytest.approx(12.0)
    assert describe(movie) == "360 frames, 12.0 s at 30 fps"
    azimuths = movie.azimuths(start_deg=20.0)
    assert azimuths[0] == pytest.approx(20.0)
    np.testing.assert_allclose(np.diff(azimuths), 1.0)
    # One step short of the full turn: the movie loops without a repeat.
    assert azimuths[-1] + 1.0 == pytest.approx(380.0)

    fast = RotationMovie(speed_deg_per_s=360.0)
    assert fast.frame_count == 30
    assert fast.seconds == pytest.approx(1.0)
    slow = RotationMovie(speed_deg_per_s=7.0)
    assert slow.frame_count == round(360 * 30 / 7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tilt_deg": 91.0},
        {"tilt_deg": -90.5},
        {"tilt_deg": float("nan")},
        {"speed_deg_per_s": 0.0},
        {"speed_deg_per_s": -10.0},
        {"fps": 0},
    ],
)
def test_a_movie_that_cannot_turn_is_refused(kwargs):
    with pytest.raises(ValueError):
        RotationMovie(**kwargs)


def test_movie_frames_are_even_rgb_and_one_size():
    first = np.arange(7 * 11 * 4, dtype=np.uint8).reshape(7, 11, 4)
    shape = movie_shape(first)
    assert shape == (6, 10)
    frame = movie_frame(first, shape)
    assert frame.shape == (6, 10, 3) and frame.dtype == np.uint8
    np.testing.assert_array_equal(frame, first[:6, :10, :3])

    # A canvas shrunk mid-recording is padded with black, not stretched.
    smaller = np.full((4, 5, 3), 200, dtype=np.uint8)
    padded = movie_frame(smaller, shape)
    assert padded.shape == (6, 10, 3)
    np.testing.assert_array_equal(padded[:4, :5], 200)
    assert not padded[4:].any() and not padded[:, 5:].any()

    grey = movie_frame(np.full((6, 10), 9, dtype=np.uint8), shape)
    np.testing.assert_array_equal(grey, 9)
    floats = movie_frame(np.full((6, 10, 3), 300.0), shape)
    assert floats.dtype == np.uint8 and int(floats.max()) == 255


def test_movie_filename_is_timestamped_and_unique(tmp_path):
    when = datetime(2026, 10, 27, 14, 5, 9)
    first = unique_movie_path(tmp_path, ".mp4", when=when)
    assert first.name == f"{ROTATION_STEM}_20261027_140509.mp4"
    first.write_bytes(b"x")
    second = unique_movie_path(tmp_path, ".mp4", when=when)
    assert second.name == f"{ROTATION_STEM}_20261027_140509_2.mp4"
    assert unique_movie_path(tmp_path, ".gif", when=when).name.endswith("140509.gif")


def _frames(count: int, shape=(8, 12)):
    return [np.full((*shape, 3), 20 * i, dtype=np.uint8) for i in range(count)]


def test_a_gif_movie_holds_every_frame_and_loops(tmp_path):
    import imageio.v2 as iio

    path = tmp_path / "turn.gif"
    writer = open_movie_writer(path, fps=25)
    for frame in _frames(5):
        writer.append_data(frame)
    writer.close()

    read = iio.mimread(path)
    assert len(read) == 5
    assert read[0].shape[:2] == (8, 12)
    assert [int(np.asarray(f)[..., :3].mean()) for f in read] == [0, 20, 40, 60, 80]
    meta = iio.get_reader(path).get_meta_data()
    assert meta["loop"] == 0
    assert meta["duration"] == pytest.approx(40, abs=10)


def test_an_mp4_movie_holds_every_frame_at_the_rate(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as iio

    path = tmp_path / "turn.mp4"
    writer = open_movie_writer(path, fps=30)
    for frame in _frames(6, shape=(10, 14)):
        writer.append_data(frame)
    writer.close()

    reader = iio.get_reader(path, format="FFMPEG")
    try:
        assert reader.count_frames() == 6
        meta = reader.get_meta_data()
        assert meta["fps"] == pytest.approx(30.0)
        # No resize to a 16-pixel macro block: the canvas's even size is kept.
        assert tuple(meta["size"]) == (14, 10)
        means = [float(np.asarray(f).mean()) for f in reader]
    finally:
        reader.close()
    np.testing.assert_allclose(means, [0, 20, 40, 60, 80, 100], atol=3)
