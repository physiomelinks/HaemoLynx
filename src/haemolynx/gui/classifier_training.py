"""What the classifier training window works with, without Qt.

Which classifier row trains for which job, the jobs it may choose between,
the layers the window paints and draws in, their colours, the plane and the
part of it the viewer shows, where a new classifier is saved, which channels
it reads, and the masks a row's classifier writes -- named as the run names
them, so a whole image segmented in the window is the run's segmentation. The
window itself is ``_widget._ClassifierTrainingWindow``; the classifier is
:mod:`haemolynx.classifier`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from haemolynx.classifier.features import DEFAULT_SCALES_UM, SCALE_CHOICES_UM
from haemolynx.classifier.model import FILE_SUFFIX, JOB_PURPOSES, PURPOSES

#: Each classifier path setting, and the job its classifier is trained for
#: (``classifier.JOB_PURPOSES``).
CLASSIFIER_ROWS: dict[str, str] = {
    "pixel_classifier_path": "vessels",
    "pixel_classifier_large_vessel_path": "vessel_classes",
    "pixel_classifier_small_vessel_path": "vessel_classes",
    "pixel_classifier_pericyte_path": "pericytes",
    "pixel_classifier_block_path": "blocks",
    "pixel_classifier_cell_path": "cells",
    "pixel_classifier_3d_vessel_path": "vessels",
}

#: Each classifier path setting's switch, turned on when a classifier is
#: saved for it from the window.
CLASSIFIER_SWITCHES: dict[str, str] = {
    "pixel_classifier_path": "use_pixel_classifier_segmentation",
    "pixel_classifier_large_vessel_path": "use_pixel_classifier_large_vessel_segmentation",
    "pixel_classifier_small_vessel_path": "use_pixel_classifier_small_vessel_segmentation",
    "pixel_classifier_pericyte_path": "use_pixel_classifier_pericyte_segmentation",
    "pixel_classifier_block_path": "use_pixel_classifier_block_segmentation",
    "pixel_classifier_cell_path": "use_pixel_classifier_cell_segmentation",
    "pixel_classifier_3d_vessel_path": "use_pixel_classifier_3d_vessel_segmentation",
}

#: What the window's title says each job finds.
JOB_TITLES: dict[str, str] = {
    "pixel_classifier_path": "vessels",
    "pixel_classifier_large_vessel_path": "large arterioles and venules",
    "pixel_classifier_small_vessel_path": "small arterioles and venules",
    "pixel_classifier_pericyte_path": "pericytes",
    "pixel_classifier_block_path": "capillary blocks",
    "pixel_classifier_cell_path": "cells",
    "pixel_classifier_3d_vessel_path": "vessels for the cell distances",
}

#: The word a new classifier's file is named with, per row.
FILE_WORDS: dict[str, str] = {
    "pixel_classifier_path": "vessels",
    "pixel_classifier_large_vessel_path": "large_vessels",
    "pixel_classifier_small_vessel_path": "small_vessels",
    "pixel_classifier_pericyte_path": "pericytes",
    "pixel_classifier_block_path": "blocks",
    "pixel_classifier_cell_path": "cells",
    "pixel_classifier_3d_vessel_path": "vessels_3d",
}

#: Each classifier purpose's name in the window's job menu.
PURPOSE_TITLES: dict[str, str] = {
    "vessels": "All vessels",
    "vessel_classes": "Arterioles and venules",
    "vessels_and_classes": "Capillaries, arterioles and venules",
    "vessels_and_sized_classes": "Capillaries, large and small arterioles and venules",
    "pericytes": "Pericytes",
    "blocks": "Capillary blocks",
    "cells": "Cells",
}

#: Each label's colour (RGBA, 0-1), by name.
LABEL_COLOURS: dict[str, tuple[float, float, float, float]] = {
    "Background": (0.62, 0.62, 0.62, 0.85),
    "Vessel": (0.95, 0.25, 0.2, 0.9),
    "Capillary": (0.2, 0.8, 0.3, 0.9),
    "Arteriole": (0.9, 0.1, 0.1, 0.9),
    "Venule": (0.15, 0.35, 0.95, 0.9),
    "Large arteriole": (0.55, 0.0, 0.0, 0.9),
    "Small arteriole": (1.0, 0.5, 0.5, 0.9),
    "Large venule": (0.0, 0.4, 0.1, 0.9),
    "Small venule": (0.65, 1.0, 0.55, 0.9),
    "Pericyte": (0.2, 0.85, 0.35, 0.9),
    "Block": (0.98, 0.65, 0.1, 0.9),
    "Cell": (0.75, 0.3, 0.9, 0.9),
}

TRANSPARENT = (0.0, 0.0, 0.0, 0.0)

#: The layers the window adds, by what they hold.
STROKES_LAYER = "Classifier strokes"
PREVIEW_LAYER = "Classifier preview"
SEGMENTATION_LAYER = "Classifier segmentation"
IMAGE_LAYER = "Classifier image"

#: Images with more voxels than this keep their strokes and preview sparse,
#: never a full-size array (as do all images under the low-RAM option).
LARGE_IMAGE_VOXELS = 1 << 28


def purposes_for(row: str) -> tuple[str, ...]:
    """The classifier purposes a row's job may be trained as."""
    return JOB_PURPOSES[CLASSIFIER_ROWS[row]]


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
    """A preview's colours: each label at its own index, the background (0)
    clear, so the image shows through."""
    colours = {0: TRANSPARENT}
    for index, name in enumerate(PURPOSES[purpose]):
        if index:
            colours[index] = LABEL_COLOURS[name]
    return colours


def mask_colours(mask: str) -> dict[int, tuple[float, float, float, float]]:
    """A 0/1 mask's colours: its label's, the background clear."""
    return {0: TRANSPARENT, 1: LABEL_COLOURS.get(mask, LABEL_COLOURS["Vessel"])}


#: Labels two jobs name differently: a stroke keeps its meaning when the job
#: changes (the combined jobs' capillaries are the vessels job's vessels).
SAME_LABEL: dict[str, str] = {"Vessel": "Capillary", "Capillary": "Vessel"}


def remap_strokes(old_purpose: str, new_purpose: str) -> dict[int, int]:
    """Stroke values (label index + 1) of *old_purpose* to *new_purpose*'s, by
    label name (or its :data:`SAME_LABEL`); a label the new purpose lacks maps
    to 0 (erased)."""
    old, new = PURPOSES[old_purpose], PURPOSES[new_purpose]

    def value(name: str) -> int:
        for candidate in (name, SAME_LABEL.get(name)):
            if candidate in new:
                return new.index(candidate) + 1
        return 0

    return {i + 1: value(name) for i, name in enumerate(old)}


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
    """The scales a new classifier starts with. Pericytes, blocks and cells
    are small objects: the finest scales, without the widest."""
    if purpose in ("pericytes", "blocks", "cells"):
        return tuple(scale for scale in SCALE_CHOICES_UM if scale <= 4.0)
    return DEFAULT_SCALES_UM


def is_large(shape: Sequence[int], values: Mapping[str, Any]) -> bool:
    """Whether the window keeps its strokes and preview sparse."""
    voxels = 1
    for n in shape:
        voxels *= int(n)
    return bool(values.get("use_memmap_loading")) or voxels > LARGE_IMAGE_VOXELS


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


def window_in_view(axis: int, corners, shape: Sequence[int]):
    """The part of a plane across *axis* on screen, ``((r0, r1), (c0, c1))``
    in the plane's own axes, from a layer's ``corner_pixels`` (data
    coordinates, ``[[min...], [max...]]``); the whole plane when unknown."""
    others = [a for a in range(3) if a != axis]
    whole = tuple((0, int(shape[a])) for a in others)
    try:
        low, high = corners[0], corners[1]
        if not any(int(high[a]) for a in others):
            return whole  # not drawn yet: napari leaves the corners at 0
        return tuple(
            (max(0, int(low[a])), min(int(shape[a]), int(high[a]) + 1)) for a in others
        )
    except (TypeError, IndexError, ValueError):
        return whole


def classifier_vessel_mask_roles(classifier_path: str | Path | None) -> tuple[str, ...]:
    """The vessel-mask roles (``"large"``, ``"small"``) the classifier at
    *classifier_path* also makes masks for (``classifier.model.VESSEL_MASK_ROLES``):
    none for a classifier of vessels alone, or a file that is not one."""
    from haemolynx.classifier import read_classifier_record
    from haemolynx.classifier.model import VESSEL_MASK_ROLES

    if classifier_path in (None, "") or not Path(classifier_path).is_file():
        return ()
    try:
        purpose = read_classifier_record(classifier_path).get("purpose")
    except (OSError, ValueError):
        return ()
    return VESSEL_MASK_ROLES.get(purpose, ())


def vessel_mask_settings_for(classifier_path: str | Path, roles: Sequence[str]) -> dict[str, Any]:
    """The settings that make *roles*' vessel masks with the classifier at
    *classifier_path*: each role's classifier row given it and its switch on
    (ilastik's off), then the Boundaries tab's vessel masks and each role's
    mask switch on. In that order, so the panel's search for masks beside the
    image finds each role made from the raw image and leaves its rows alone."""
    from haemolynx.io.vessel_masks import VESSEL_MASK_SETTINGS

    settings: dict[str, Any] = {}
    for role in roles:
        names = VESSEL_MASK_SETTINGS[role]
        settings[names["classifier_path"]] = str(classifier_path)
        settings[names["use_ilastik"]] = False
        settings[names["use_classifier"]] = True
    if roles:
        settings["automated_vessel_assignment"] = True
    for role in roles:
        settings[VESSEL_MASK_SETTINGS[role]["enabled"]] = True
    return settings


def vessel_mask_note(roles: Sequence[str]) -> str:
    """What the panel says when a classifier fills *roles*' vessel masks."""
    sizes = " and ".join(roles)
    return (
        f"The classifier also finds {sizes} arterioles and venules: the Boundaries tab's "
        f"{sizes} vessel masks are now made with it."
    )


def row_outputs(values: Mapping[str, Any], row: str, classifier_path: str | Path) -> dict[str, Path]:
    """The masks a row's classifier writes, ``{mask: path}``, named as the
    run names them (``stages.classifier_requests``), so the run finds them
    current and uses them."""
    from haemolynx.io.vessel_masks import classifier_role_outputs
    from haemolynx.pipeline.naming import classifier_segmented_path
    from haemolynx.pipeline.stages import object_mask_output_path

    if row == "pixel_classifier_path":
        return {"Vessel": classifier_segmented_path(values)}
    if row in ("pixel_classifier_large_vessel_path", "pixel_classifier_small_vessel_path"):
        role = "large" if "large" in row else "small"
        return classifier_role_outputs(values.get("output_dir"), values["raw_image_path"], role, classifier_path)
    word, mask = {
        "pixel_classifier_pericyte_path": ("pericytes", "Pericyte"),
        "pixel_classifier_block_path": ("blocks", "Block"),
        "pixel_classifier_cell_path": ("cells", "Cell"),
        "pixel_classifier_3d_vessel_path": ("vessels_3d", "Vessel"),
    }[row]
    return {mask: object_mask_output_path(values, word)}
