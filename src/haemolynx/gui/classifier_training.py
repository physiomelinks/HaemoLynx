"""What the classifier training window works with, without Qt.

Which classifier row trains for which job, the layers the window paints and
draws in, their colours, the plane the viewer shows, where a new classifier
is saved, and which channels it reads. The window itself is
``_widget._ClassifierTrainingWindow``; the classifier is
:mod:`haemolynx.classifier`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from haemolynx.classifier.features import DEFAULT_SCALES_UM, SCALE_CHOICES_UM
from haemolynx.classifier.model import FILE_SUFFIX, PURPOSES

#: Each classifier path setting, and the job its classifier is trained for.
CLASSIFIER_ROWS: dict[str, str] = {
    "pixel_classifier_path": "vessels",
    "pixel_classifier_large_vessel_path": "vessel_classes",
    "pixel_classifier_small_vessel_path": "vessel_classes",
    "pixel_classifier_pericyte_path": "pericytes",
    "pixel_classifier_block_path": "blocks",
}

#: Each classifier path setting's switch, turned on when a classifier is
#: saved for it from the window.
CLASSIFIER_SWITCHES: dict[str, str] = {
    "pixel_classifier_path": "use_pixel_classifier_segmentation",
    "pixel_classifier_large_vessel_path": "use_pixel_classifier_large_vessel_segmentation",
    "pixel_classifier_small_vessel_path": "use_pixel_classifier_small_vessel_segmentation",
    "pixel_classifier_pericyte_path": "use_pixel_classifier_pericyte_segmentation",
    "pixel_classifier_block_path": "use_pixel_classifier_block_segmentation",
}

#: What the window's title says each job finds.
JOB_TITLES: dict[str, str] = {
    "pixel_classifier_path": "vessels",
    "pixel_classifier_large_vessel_path": "large arterioles and venules",
    "pixel_classifier_small_vessel_path": "small arterioles and venules",
    "pixel_classifier_pericyte_path": "pericytes",
    "pixel_classifier_block_path": "capillary blocks",
}

#: The word a new classifier's file is named with, per row.
FILE_WORDS: dict[str, str] = {
    "pixel_classifier_path": "vessels",
    "pixel_classifier_large_vessel_path": "large_vessels",
    "pixel_classifier_small_vessel_path": "small_vessels",
    "pixel_classifier_pericyte_path": "pericytes",
    "pixel_classifier_block_path": "blocks",
}

#: Each label's colour (RGBA, 0-1), by name.
LABEL_COLOURS: dict[str, tuple[float, float, float, float]] = {
    "Background": (0.62, 0.62, 0.62, 0.85),
    "Vessel": (0.95, 0.25, 0.2, 0.9),
    "Arteriole": (0.95, 0.25, 0.2, 0.9),
    "Venule": (0.25, 0.45, 0.95, 0.9),
    "Pericyte": (0.2, 0.85, 0.35, 0.9),
    "Block": (0.98, 0.65, 0.1, 0.9),
}

TRANSPARENT = (0.0, 0.0, 0.0, 0.0)

#: The layers the window adds, by what they hold.
STROKES_LAYER = "Classifier strokes"
PREVIEW_LAYER = "Classifier preview"
SEGMENTATION_LAYER = "Classifier segmentation"
IMAGE_LAYER = "Classifier image"


def layer_name(kind: str, row: str) -> str:
    """``Classifier strokes (vessels)``: one set of layers per job."""
    return f"{kind} ({JOB_TITLES[row]})"


def stroke_colours(purpose: str) -> dict[int, tuple[float, float, float, float]]:
    """The strokes layer's colours: 0 unpainted, each label at its index + 1."""
    colours = {0: TRANSPARENT}
    for index, name in enumerate(PURPOSES[purpose]):
        colours[index + 1] = LABEL_COLOURS[name]
    return colours


def result_colours(purpose: str) -> dict[int, tuple[float, float, float, float]]:
    """A preview's or segmentation's colours: each label at its own index,
    the background (0) clear, so the image shows through."""
    colours = {0: TRANSPARENT}
    for index, name in enumerate(PURPOSES[purpose]):
        if index:
            colours[index] = LABEL_COLOURS[name]
    return colours


def default_classifier_path(raw_image_path: str | Path, row: str) -> Path:
    """``{raw image folder}/{raw stem}_{job}.haemoclf``: beside the image it
    was trained on, named for its job."""
    raw = Path(raw_image_path)
    return raw.with_name(f"{raw.stem}_{FILE_WORDS[row]}{FILE_SUFFIX}")


def default_channels(values: Mapping[str, Any], channel_count: int) -> tuple[int | None, ...]:
    """The channels a new classifier reads: none for a one-channel image;
    otherwise the run's raw image channel, else the first."""
    if channel_count <= 1:
        return (None,)
    channel = values.get("raw_image_channel")
    try:
        channel = int(channel)
    except (TypeError, ValueError):
        channel = 0
    return (channel if 0 <= channel < channel_count else 0,)


def default_scales(purpose: str) -> tuple[float, ...]:
    """The scales a new classifier starts with. Pericytes and blocks are
    small objects: the finest scales, without the widest."""
    if purpose in ("pericytes", "blocks"):
        return tuple(scale for scale in SCALE_CHOICES_UM if scale <= 4.0)
    return DEFAULT_SCALES_UM


def plane_in_view(displayed: Sequence[int], point: Sequence[float], shape: Sequence[int]) -> tuple[int, int] | None:
    """``(axis, index)`` of the plane the viewer shows of a 3D volume of
    *shape*: the axis not displayed, at *point* (data coordinates) rounded.
    None in 3D view, or for a point outside the volume."""
    displayed = [int(axis) for axis in displayed][-2:]
    if len(displayed) != 2 or len(shape) != 3:
        return None
    hidden = [axis for axis in range(3) if axis not in displayed]
    if len(hidden) != 1:
        return None
    axis = hidden[0]
    index = int(round(float(point[axis])))
    if not 0 <= index < int(shape[axis]):
        return None
    return axis, index
