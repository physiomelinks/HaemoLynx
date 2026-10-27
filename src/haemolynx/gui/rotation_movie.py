"""A movie of the canvas turning one full circle, for the view window.

The camera turns about the volume's z axis, the stack axis, which stands up
the screen with the top slice at the top, as in the XZ and YZ views. The
tilt is how far the view looks down on the stack from above while it turns:
0 degrees is side on, 90 straight down onto the top slice, and a negative
tilt looks up from below. The turn starts where the camera already looks
round the axis and keeps its zoom and centre, so the first frame is the view
on screen seen at the chosen tilt. Rotate turns it the same way on screen,
writing nothing, until stopped.

Pure up to :func:`rotation_box`: camera directions, frame counts, frame
fitting and the movie file. ``rotation_box`` builds the view window's box,
turns and records with Qt, importing it only when called.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Frames a second the movie plays at.
MOVIE_FPS = 30

DEFAULT_TILT_DEG = 30.0
DEFAULT_SPEED_DEG_PER_S = 30.0
MAX_TILT_DEG = 90.0
MIN_SPEED_DEG_PER_S = 1.0
MAX_SPEED_DEG_PER_S = 360.0

#: ``haemolynx_rotation_YYYYMMDD_HHMMSS.mp4`` in the pipeline output folder.
ROTATION_STEM = "haemolynx_rotation"
ROTATION_NO_OUTPUT_FOLDER = (
    "No output folder is set; set the VTK output prefix before recording a rotation."
)
ROTATION_NO_VIEWER = "No viewer to rotate or record."
ROTATION_NOTHING_SHOWN = "Nothing on the canvas to rotate or record."
RECORD_LABEL = "Record 360° rotation"
ROTATE_LABEL = "Rotate"
STOP_ROTATING_LABEL = "Stop rotating"

#: The stack axis, canonical ``(z, y, x)``.
_AXIS = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class RotationMovie:
    """One full turn at *speed_deg_per_s*, seen at *tilt_deg*, played at *fps*."""

    tilt_deg: float = DEFAULT_TILT_DEG
    speed_deg_per_s: float = DEFAULT_SPEED_DEG_PER_S
    fps: int = MOVIE_FPS

    def __post_init__(self) -> None:
        if not abs(float(self.tilt_deg)) <= MAX_TILT_DEG:
            raise ValueError(f"tilt must be within ±{MAX_TILT_DEG:g}°, got {self.tilt_deg}")
        if not float(self.speed_deg_per_s) > 0:
            raise ValueError(f"speed must be above 0°/s, got {self.speed_deg_per_s}")
        if int(self.fps) < 1:
            raise ValueError(f"frame rate must be at least 1, got {self.fps}")

    @property
    def frame_count(self) -> int:
        """Frames in one turn; the last stops a step short of the first, so
        the movie loops without a repeated frame."""
        return max(2, round(360.0 * self.fps / float(self.speed_deg_per_s)))

    @property
    def seconds(self) -> float:
        return self.frame_count / self.fps

    def azimuths(self, start_deg: float = 0.0) -> np.ndarray:
        """Each frame's angle round the axis, in degrees from *start_deg*."""
        steps = np.arange(self.frame_count, dtype=float)
        return float(start_deg) + 360.0 * steps / self.frame_count


def describe(movie: RotationMovie) -> str:
    """What the box shows under its controls."""
    return f"{movie.frame_count} frames, {movie.seconds:.1f} s at {movie.fps} fps"


def turntable_directions(
    azimuth_deg: float, tilt_deg: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(view_direction, up_direction)`` in canonical ``(z, y, x)`` for one
    frame, for napari's ``Camera.set_view_direction``.

    Azimuth 0 at tilt 0 is the XZ view and azimuth 90 the YZ view; tilt -90
    at azimuth 0 is the XY view (``view_snap.camera_directions_for_plane``).
    Up is always the z axis's own lean on screen, so the axis stays upright.
    """
    phi = math.radians(float(azimuth_deg))
    tilt = math.radians(float(tilt_deg))
    across = np.array([0.0, math.cos(phi), -math.sin(phi)])
    view = math.cos(tilt) * across + math.sin(tilt) * _AXIS
    up = -math.cos(tilt) * _AXIS + math.sin(tilt) * across
    return _as_tuple(view), _as_tuple(up)


def start_azimuth(view: Sequence[float], up: Sequence[float]) -> float:
    """The azimuth, in degrees, that a camera looking along *view* with *up*
    on screen is turned to round the z axis: :func:`turntable_directions`'
    inverse for any of its tilts, and 0 when the camera gives no turn."""
    view = np.asarray(view, dtype=float)
    up = np.asarray(up, dtype=float)
    # Across the axis, view leans cos(tilt) and up sin(tilt) times the
    # azimuth's direction; this sum is a positive multiple of it at every tilt.
    across = view[1:] + view[0] * up[1:]
    if float(np.hypot(*across)) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(-across[1], across[0])) % 360.0


def displayed_axes(displayed: Sequence[int], ndim: int) -> tuple[int, int, int]:
    """Which canonical ``(z, y, x)`` axis each displayed dim is, so a 3D view
    whose dims were reordered still turns about z. Identity where the
    displayed dims are not the volume's three spatial ones."""
    offset = int(ndim) - 3
    axes = tuple(int(dim) - offset for dim in displayed)
    return axes if sorted(axes) == [0, 1, 2] else (0, 1, 2)


def to_displayed(vector_zyx: Sequence[float], axes: Sequence[int]) -> tuple[float, float, float]:
    """*vector_zyx* in the displayed dims' order."""
    return tuple(float(vector_zyx[axis]) for axis in axes)


def from_displayed(vector: Sequence[float], axes: Sequence[int]) -> tuple[float, float, float]:
    """A displayed-order vector back in canonical ``(z, y, x)``."""
    canonical = [0.0, 0.0, 0.0]
    for value, axis in zip(vector, axes):
        canonical[axis] = float(value)
    return tuple(canonical)


def movie_shape(frame: np.ndarray) -> tuple[int, int]:
    """Rows and columns of a movie whose first frame is *frame*: rounded
    down to even, which H.264 needs."""
    rows, columns = int(frame.shape[0]), int(frame.shape[1])
    return max(2, rows - rows % 2), max(2, columns - columns % 2)


def movie_frame(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """*frame* as one RGB ``uint8`` frame of *shape*: alpha dropped, cropped
    at the bottom and right or padded there with black, so a canvas resized
    mid-recording cannot change the movie's size."""
    rgb = np.asarray(frame)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rows, columns = shape
    out = np.zeros((rows, columns, 3), dtype=np.uint8)
    kept = rgb[:rows, :columns]
    out[: kept.shape[0], : kept.shape[1]] = kept
    return out


def movie_suffix() -> str:
    """``.mp4`` where imageio can reach ffmpeg (the ``imageio-ffmpeg``
    package), else ``.gif``."""
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return ".gif"
    return ".mp4"


def unique_movie_path(
    output_dir: Path, suffix: str = ".mp4", *, when: datetime | None = None
) -> Path:
    """``haemolynx_rotation_YYYYMMDD_HHMMSS.mp4``; ``_2``, ``_3``, … if taken."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate = Path(output_dir) / f"{ROTATION_STEM}_{stamp}{suffix}"
    n = 2
    while candidate.exists():
        candidate = Path(output_dir) / f"{ROTATION_STEM}_{stamp}_{n}{suffix}"
        n += 1
    return candidate


def open_movie_writer(path: Path, fps: int = MOVIE_FPS):
    """An imageio writer for *path*, by its suffix: H.264 for ``.mp4``,
    looping for ``.gif``. Frames go in with ``append_data``."""
    import imageio.v2 as iio

    path = Path(path)
    if path.suffix.lower() == ".gif":
        return iio.get_writer(str(path), mode="I", duration=1000.0 / fps, loop=0)
    return iio.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=2,
        ffmpeg_log_level="error",
    )


def viewer_camera(viewer):
    """*viewer*'s camera: ``viewer.scene.camera`` from napari 0.9, which
    deprecated ``viewer.camera``."""
    camera = getattr(getattr(viewer, "scene", None), "camera", None)
    return camera if camera is not None else viewer.camera


def _as_tuple(vector: np.ndarray) -> tuple[float, float, float]:
    # + 0.0 turns -0.0 into 0.0, so a direction compares equal to a snap's.
    return tuple(float(value) + 0.0 for value in vector)


@dataclass
class _CameraState:
    ndisplay: int
    angles: tuple
    center: tuple
    zoom: float
    point: tuple

    @classmethod
    def of(cls, viewer) -> "_CameraState":
        camera = viewer_camera(viewer)
        return cls(
            ndisplay=int(viewer.dims.ndisplay),
            angles=tuple(camera.angles),
            center=tuple(camera.center),
            zoom=float(camera.zoom),
            point=tuple(viewer.dims.point),
        )

    def restore(self, viewer, *, framing: bool = True) -> None:
        """Put the view back; without *framing*, keep the zoom and centre
        it has now, as a pan or zoom made while Rotate turned it left them."""
        if int(viewer.dims.ndisplay) != self.ndisplay:
            viewer.dims.ndisplay = self.ndisplay
        camera = viewer_camera(viewer)
        camera.angles = self.angles
        if framing:
            camera.center = self.center
            camera.zoom = self.zoom
        if tuple(viewer.dims.point) != self.point:
            viewer.dims.point = self.point


def rotation_box(
    viewer,
    *,
    output_folder: Callable[[], Path | None],
    screenshot: Callable[[], Any],
    report: Callable[[str], None],
    clock: Callable[[], float] = time.monotonic,
):
    """The view window's Rotation movie group, turning *viewer*'s canvas.

    Rotate turns the camera at the chosen tilt and speed, by *clock*'s
    seconds, until pressed again, reading both afresh each tick so they can
    be tried while it turns; it writes nothing, and on stopping the view
    turns back, keeping any zoom or pan made meanwhile. Record stops it
    first, then turns the camera one frame a tick of the Qt event loop, so
    the canvas is seen turning and Stop stays live; each frame is
    *screenshot*'s picture of the canvas, written straight to the movie.
    Either turns a 2D view in 3D. After recording, the view, camera and
    slice are put back as they were, and *report* is told where the movie
    went; a stopped or failed recording leaves no file.
    """
    from types import SimpleNamespace

    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import (
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QLabel,
        QPushButton,
    )

    from haemolynx.gui.chrome_tooltips import ROTATION_TOOLTIPS

    group = QGroupBox("Rotation movie")
    group.setObjectName("haemolynx_rotation_group")
    form = QFormLayout(group)

    def spin(name: str, low: float, high: float, value: float, step: float, suffix: str):
        widget = QDoubleSpinBox()
        widget.setObjectName(f"haemolynx_rotation_{name}")
        widget.setRange(low, high)
        widget.setDecimals(0)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setSuffix(suffix)
        widget.setToolTip(ROTATION_TOOLTIPS[name])
        return widget

    tilt = spin("tilt", -MAX_TILT_DEG, MAX_TILT_DEG, DEFAULT_TILT_DEG, 5.0, " °")
    speed = spin(
        "speed", MIN_SPEED_DEG_PER_S, MAX_SPEED_DEG_PER_S, DEFAULT_SPEED_DEG_PER_S, 5.0, " °/s"
    )
    summary = QLabel()
    summary.setObjectName("haemolynx_rotation_summary")
    rotate = QPushButton(ROTATE_LABEL)
    rotate.setObjectName("haemolynx_rotation_rotate")
    rotate.setToolTip(ROTATION_TOOLTIPS["rotate"])
    record = QPushButton(RECORD_LABEL)
    record.setObjectName("haemolynx_rotation_record")
    record.setToolTip(ROTATION_TOOLTIPS["record"])

    def add(label: str, widget) -> None:
        caption = QLabel(label)
        caption.setToolTip(widget.toolTip())
        form.addRow(caption, widget)

    add("Axis tilt", tilt)
    add("Speed", speed)
    form.addRow(summary)
    form.addRow(rotate)
    form.addRow(record)

    timer = QTimer(group)
    timer.setInterval(0)
    turn_timer = QTimer(group)
    turn_timer.setInterval(round(1000 / MOVIE_FPS))

    box = SimpleNamespace(
        group=group,
        tilt=tilt,
        speed=speed,
        summary=summary,
        rotate=rotate,
        record=record,
        timer=timer,
        turn_timer=turn_timer,
        rotating=False,
        recording=False,
        path=None,
    )
    take: dict[str, Any] = {}
    turning: dict[str, Any] = {}

    def movie() -> RotationMovie:
        return RotationMovie(tilt_deg=tilt.value(), speed_deg_per_s=speed.value())

    def refresh_summary(*_args) -> None:
        summary.setText(describe(movie()))

    def set_recording(on: bool) -> None:
        box.recording = on
        tilt.setEnabled(not on)
        speed.setEnabled(not on)
        rotate.setEnabled(not on)
        if not on:
            record.setText(RECORD_LABEL)

    def take_view(action: str) -> dict[str, Any] | None:
        """Remember the view, show it in 3D and find where the camera looks
        round z; None, with the reason reported, when it cannot turn."""
        if viewer is None:
            report(ROTATION_NO_VIEWER)
            return None
        if not any(getattr(layer, "visible", True) for layer in viewer.layers):
            report(ROTATION_NOTHING_SHOWN)
            return None
        state = _CameraState.of(viewer)
        try:
            if state.ndisplay != 3:
                viewer.dims.ndisplay = 3
            axes = displayed_axes(viewer.dims.displayed, viewer.dims.ndim)
            camera = viewer_camera(viewer)
            start_deg = start_azimuth(
                from_displayed(camera.view_direction, axes),
                from_displayed(camera.up_direction, axes),
            )
        except Exception as error:  # noqa: BLE001 - must not crash Qt
            logger.exception("could not start to %s the view", action)
            state.restore(viewer)
            report(f"Could not {action} the rotation:\n{error}")
            return None
        return {"state": state, "axes": axes, "azimuth": start_deg}

    def turn_to(azimuth: float, tilt_deg: float, axes) -> None:
        view, up = turntable_directions(azimuth, tilt_deg)
        viewer_camera(viewer).set_view_direction(
            to_displayed(view, axes), to_displayed(up, axes)
        )

    def stop_turning() -> None:
        if not box.rotating:
            return
        turn_timer.stop()
        state = turning.get("state")
        turning.clear()
        box.rotating = False
        rotate.setText(ROTATE_LABEL)
        if state is not None:
            try:
                state.restore(viewer, framing=False)
            except Exception:  # noqa: BLE001 - must not crash Qt
                logger.debug("could not turn the view back after rotating", exc_info=True)

    def turn_on() -> None:
        now = clock()
        turning["azimuth"] = (
            turning["azimuth"] + speed.value() * (now - turning["last"])
        ) % 360.0
        turning["last"] = now
        try:
            turn_to(turning["azimuth"], tilt.value(), turning["axes"])
        except Exception as error:  # noqa: BLE001 - must not crash Qt
            logger.exception("could not rotate the view")
            stop_turning()
            report(f"Could not rotate the view:\n{error}")

    def start_turning() -> None:
        if box.rotating or box.recording:
            return
        taken = take_view("rotate")
        if taken is None:
            return
        turning.update(taken, last=clock())
        box.rotating = True
        rotate.setText(STOP_ROTATING_LABEL)
        try:
            turn_to(turning["azimuth"], tilt.value(), turning["axes"])
        except Exception as error:  # noqa: BLE001 - must not crash Qt
            logger.exception("could not rotate the view")
            stop_turning()
            report(f"Could not rotate the view:\n{error}")
            return
        turn_timer.start()

    def on_rotate() -> None:
        if box.rotating:
            stop_turning()
        else:
            start_turning()

    def finish(message: str, *, keep: bool) -> None:
        timer.stop()
        writer = take.pop("writer", None)
        path = take.pop("path", None)
        state = take.pop("state", None)
        take.clear()
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - the file is dropped below
                logger.debug("could not close the rotation movie", exc_info=True)
                keep = False
        if not keep and path is not None:
            Path(path).unlink(missing_ok=True)
        if state is not None:
            try:
                state.restore(viewer)
            except Exception:  # noqa: BLE001 - must not crash Qt
                logger.debug("could not put the view back after recording", exc_info=True)
        box.path = Path(path) if keep and path is not None else None
        set_recording(False)
        report(message)

    def step() -> None:
        index = take["index"]
        azimuths = take["azimuths"]
        try:
            turn_to(azimuths[index], take["tilt"], take["axes"])
            frame = screenshot()
            if not box.recording:
                # Stopped while the canvas was being captured.
                return
            if frame is None:
                raise ValueError("Could not capture the canvas.")
            frame = np.asarray(frame)
            if "writer" not in take:
                take["shape"] = movie_shape(frame)
                take["writer"] = open_movie_writer(take["path"], take["fps"])
            take["writer"].append_data(movie_frame(frame, take["shape"]))
        except Exception as error:  # noqa: BLE001 - must not crash Qt
            logger.exception("could not record the rotation")
            finish(f"Could not record the rotation:\n{error}", keep=False)
            return
        take["index"] = index + 1
        if take["index"] >= len(azimuths):
            path = take["path"]
            finish(f"Wrote rotation movie {path}", keep=True)
            return
        record.setText(f"Stop recording ({take['index']}/{len(azimuths)})")

    def start() -> None:
        if viewer is None:
            report(ROTATION_NO_VIEWER)
            return
        folder = output_folder()
        if folder is None:
            report(ROTATION_NO_OUTPUT_FOLDER)
            return
        # From the view Rotate turns back to, with any zoom or pan made
        # while it turned.
        stop_turning()
        taken = take_view("record")
        if taken is None:
            return
        try:
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            taken["state"].restore(viewer)
            report(f"Could not record the rotation:\n{error}")
            return
        chosen = movie()
        take.update(
            state=taken["state"],
            axes=taken["axes"],
            tilt=chosen.tilt_deg,
            fps=chosen.fps,
            azimuths=chosen.azimuths(taken["azimuth"]),
            index=0,
            path=unique_movie_path(folder, movie_suffix()),
        )
        set_recording(True)
        record.setText(f"Stop recording (0/{chosen.frame_count})")
        timer.start()

    def stop() -> None:
        if box.recording:
            finish("Stopped recording the rotation; no movie written.", keep=False)

    def on_record() -> None:
        if box.recording:
            stop()
        else:
            start()

    timer.timeout.connect(step)
    turn_timer.timeout.connect(turn_on)
    rotate.clicked.connect(on_rotate)
    record.clicked.connect(on_record)
    tilt.valueChanged.connect(refresh_summary)
    speed.valueChanged.connect(refresh_summary)
    refresh_summary()
    box.movie = movie
    box.start = start
    box.stop = stop
    box.start_turning = start_turning
    box.stop_turning = stop_turning
    return box
