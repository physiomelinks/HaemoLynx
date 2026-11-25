"""The edge-case suite (``robust.6a``): small masks that bad or unusual input
looks like, each run through every stage with the schema's defaults.

A case passes when the run finishes, or stops with a named error: an error
HaemoLynx raised on purpose to say what is wrong with the input. An
unhandled exception -- an IndexError from deep in numpy, a KeyError, a
ZeroDivisionError -- fails it, as would a wrong answer it never noticed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

#: Exception types that mean a programming error wherever they come from,
#: never a deliberate stop.
PROGRAMMING_ERRORS = frozenset({
    "IndexError", "KeyError", "TypeError", "AttributeError", "ZeroDivisionError",
    "AssertionError", "NameError", "UnboundLocalError", "RecursionError", "StopIteration",
    "NotImplementedError", "FloatingPointError", "OverflowError",
})


def classify_failure(error_type: Optional[str], raised_in: Optional[str]) -> str:
    """``finished`` when nothing was raised; ``named`` when the error is
    HaemoLynx's own (raised from a ``haemolynx`` module) and not a
    programming error; ``unhandled`` otherwise -- including a ValueError
    raised inside numpy or scipy, which HaemoLynx did not catch and name."""
    if error_type is None:
        return "finished"
    if error_type in PROGRAMMING_ERRORS:
        return "unhandled"
    if raised_in and (raised_in == "haemolynx" or raised_in.startswith("haemolynx.")):
        return "named"
    return "unhandled"


@dataclass(frozen=True)
class EdgeCase:
    name: str
    description: str
    make: Callable[[], np.ndarray]
    #: Settings beside the defaults: the voxel size, mostly.
    settings: dict[str, Any] = field(default_factory=dict)


SHAPE = (24, 40, 40)


def _empty() -> np.ndarray:
    return np.zeros(SHAPE, dtype=np.uint8)


def _tube(radius: float, *, through: bool) -> np.ndarray:
    """A straight tube along x at the volume's middle, from face to face when
    *through*, else ending 8 voxels inside each."""
    volume = np.zeros(SHAPE, dtype=np.uint8)
    z, y = np.ogrid[: SHAPE[0], : SHAPE[1]]
    section = (z - SHAPE[0] // 2) ** 2 + (y - SHAPE[1] // 2) ** 2 <= radius ** 2
    start, stop = (0, SHAPE[2]) if through else (8, SHAPE[2] - 8)
    volume[:, :, start:stop] = section[:, :, None]
    return volume


def _one_vessel() -> np.ndarray:
    return _tube(3.0, through=False)


def _one_voxel_thick() -> np.ndarray:
    volume = np.zeros(SHAPE, dtype=np.uint8)
    volume[SHAPE[0] // 2, SHAPE[1] // 2, 4:-4] = 1
    return volume


def _touching_border() -> np.ndarray:
    """A T: a tube from face to face along x and a branch running out of the
    top face, so every end is cut by the image."""
    volume = _tube(3.0, through=True)
    z, x = np.ogrid[: SHAPE[0], : SHAPE[2]]
    branch = (z - SHAPE[0] // 2) ** 2 + (x - SHAPE[2] // 2) ** 2 <= 9.0
    volume[:, : SHAPE[1] // 2, :] |= branch[:, None, :].astype(np.uint8)
    return volume


def _closed_ring() -> np.ndarray:
    """A torus wholly inside the volume: vessels, but no end at any face to
    be an inlet or an outlet."""
    z, y, x = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    centre = np.array(SHAPE) / 2.0 - 0.5
    planar = np.sqrt((y - centre[1]) ** 2 + (x - centre[2]) ** 2)
    return (((planar - 12.0) ** 2 + (z - centre[0]) ** 2) <= 9.0).astype(np.uint8)


def _two_dimensional() -> np.ndarray:
    """A 2D image: a cross of two lines three pixels wide."""
    image = np.zeros(SHAPE[1:], dtype=np.uint8)
    middle = SHAPE[1] // 2
    image[middle - 1 : middle + 2, 2:-2] = 1
    image[2:-2, middle - 1 : middle + 2] = 1
    return image


CASES: tuple[EdgeCase, ...] = (
    EdgeCase("empty_mask", "a mask with no vessel at all", _empty),
    EdgeCase("one_vessel", "one straight vessel ending inside the image", _one_vessel),
    EdgeCase("one_voxel_thick", "a vessel one voxel thick", _one_voxel_thick),
    EdgeCase("two_dimensional", "a 2D image", _two_dimensional),
    EdgeCase(
        "anisotropic", "one vessel on voxels 0.3 x 0.3 x 5 um (x, y, z)", _one_vessel,
        {"voxel_size_override_xyz": [0.3, 0.3, 5.0], "voxel_size_policy": "override"},
    ),
    EdgeCase("touching_border", "vessels cut by the image faces", _touching_border),
    EdgeCase("no_inlets_or_outlets", "a closed ring touching no face", _closed_ring),
)

CASES_BY_NAME = {case.name: case for case in CASES}
