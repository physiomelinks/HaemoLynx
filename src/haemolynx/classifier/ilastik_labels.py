"""The labels painted in an ilastik project, read as training strokes.

An ilastik Pixel Classification project (``.ilp``, HDF5) keeps its label
names in ``PixelClassification/LabelNames`` and each image's painted voxels
in ``PixelClassification/LabelSets/labelsNNN``: blocks, each a small array of
label numbers (1 the first name, 0 unpainted) with ``blockSlice`` saying where
it lies and ``axistags`` naming its axes. The images are listed under
``Input Data/infos/laneNNNN/Raw Data`` with their file, axes and shape.

:func:`read_ilastik_labels` gives the painted voxels of the image that is the
raw image (by file name, or the project's only image), in the raw image's
canonical ``(z, y, x)`` as HaemoLynx reads it, after checking its shape;
:func:`default_label_mapping` says which of a classifier's labels each ilastik
label most likely is. The project is only read.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Mapping, Sequence

import numpy as np

#: Words read as another label's name.
_SYNONYMS = {
    "bg": "background", "back ground": "background", "backgrounds": "background", "tissue": "background",
    "vessels": "vessel", "artery": "arteriole", "arteries": "arteriole", "vein": "venule", "veins": "venule",
}


@dataclass(frozen=True)
class IlastikImage:
    """One image of an ilastik project, as the project records it."""

    lane: int
    file_name: str
    axes: str  # the image's axes as ilastik tags them, e.g. "zcyx"
    shape: tuple[int, ...]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Its z, y and x sizes (y and x for a 2D image), in its own order."""
        return tuple(n for key, n in zip(self.axes, self.shape) if key in "zyx")


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _text(value[()])
    return str(value)


def _axis_keys(axistags) -> str:
    """``"zyxc"`` from ilastik's axistags JSON."""
    try:
        return "".join(axis["key"] for axis in json.loads(_text(axistags))["axes"])
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot read the axes ilastik recorded ({error}).") from error


def ilastik_label_names(project: str | Path) -> tuple[str, ...]:
    """The project's label names, in ilastik's order (label 1 first), as the
    run reads them (``io.ilastik.ilastik_label_names``)."""
    from haemolynx.io.ilastik import ilastik_label_names as read_label_names

    names = read_label_names(project)
    if names is None:
        raise ValueError(f"{Path(project).name} is not an ilastik Pixel Classification project with labels.")
    return tuple(names)


def ilastik_images(project: str | Path) -> list[IlastikImage]:
    """The images the project was trained on."""
    import h5py

    images = []
    with h5py.File(str(project), "r") as f:
        infos = f.get("Input Data/infos")
        if infos is None:
            return images
        for lane in sorted(infos):
            raw = infos[lane].get("Raw Data")
            if raw is None or "filePath" not in raw:
                continue
            path = _text(raw["filePath"][()])
            axes = _axis_keys(raw["axistags"][()]) if "axistags" in raw else ""
            shape = tuple(int(n) for n in raw["shape"][()]) if "shape" in raw else ()
            images.append(IlastikImage(
                lane=int(re.sub(r"\D", "", lane) or 0),
                # ilastik on Windows writes backslashes, elsewhere slashes.
                file_name=PureWindowsPath(path.replace("/", "\\")).name,
                axes=axes, shape=shape,
            ))
    return images


def _image_for(project: Path, raw_image_path: Path) -> IlastikImage:
    images = ilastik_images(project)
    named = [image for image in images if image.file_name.lower() == raw_image_path.name.lower()]
    if len(named) == 1:
        return named[0]
    if len(images) == 1 and not named:
        return images[0]
    listed = ", ".join(image.file_name for image in images) or "none"
    raise ValueError(
        f"{project.name} has no labels for {raw_image_path.name} alone; its images are: {listed}."
    )


def _block_slices(text: str) -> list[slice]:
    """``[31:32,157:160,0:1]`` as slices."""
    parts = text.strip().strip("[]").split(",")
    slices = []
    for part in parts:
        start, _, stop = part.partition(":")
        slices.append(slice(int(start), int(stop)))
    return slices


def read_ilastik_labels(
    project: str | Path,
    raw_image_path: str | Path,
    *,
    shape_zyx: Sequence[int],
    axis_order: str = "zyx",
) -> tuple[np.ndarray, np.ndarray]:
    """``(points, labels)``: every voxel painted in the project's image that
    is the raw image -- ``points`` ``(n, 3)`` int64 in the raw image's
    canonical ``(z, y, x)`` (*shape_zyx*, read in *axis_order*), ``labels``
    the ilastik label numbers (1 its first name). Raises ``ValueError`` when
    the project has no such image, or its image is another size."""
    import h5py

    from haemolynx.io.axis_order import axis_order_transpose, normalize_axis_order

    project, raw_image_path = Path(project), Path(raw_image_path)
    image = _image_for(project, raw_image_path)
    shape_zyx = tuple(int(n) for n in shape_zyx)
    order = normalize_axis_order(axis_order)
    to_canonical = axis_order_transpose(order)
    stored_shape = tuple(int(shape_zyx[to_canonical.index(axis)]) for axis in range(3))
    spatial = [key for key in image.axes if key in "zyx"] or ["z", "y", "x"]
    two_d = len(spatial) == 2
    # A one-plane image is read as it is stored, whatever the axis order.
    expected = shape_zyx[1:] if two_d else stored_shape
    if image.spatial_shape and (
        (two_d and shape_zyx[0] != 1) or tuple(image.spatial_shape) != tuple(expected)
    ):
        raise ValueError(
            f"{project.name}'s image {image.file_name} is {' x '.join(map(str, image.spatial_shape))} voxels; "
            f"{raw_image_path.name} is {' x '.join(map(str, expected))}."
        )
    points: list[np.ndarray] = []
    values: list[np.ndarray] = []
    with h5py.File(str(project), "r") as f:
        labels = f.get(f"PixelClassification/LabelSets/labels{image.lane:03d}")
        if labels is None:
            return np.zeros((0, 3), np.int64), np.zeros(0, np.uint8)
        for name in sorted(labels):
            block = labels[name]
            keys = _axis_keys(block.attrs["axistags"]) if "axistags" in block.attrs else "zyxc"[-block.ndim:]
            slices = _block_slices(_text(block.attrs["blockSlice"]))
            data = np.asarray(block[()])
            found = np.nonzero(data)
            if not len(found[0]):
                continue
            position = {key: found[i] + slices[i].start for i, key in enumerate(keys)}
            stored = np.zeros((len(found[0]), 3), np.int64)
            for column, key in enumerate(spatial):
                stored[:, column + (1 if two_d else 0)] = position[key]
            points.append(stored[:, list(to_canonical)] if not two_d else stored)
            values.append(data[found].astype(np.uint8))
    if not points:
        return np.zeros((0, 3), np.int64), np.zeros(0, np.uint8)
    points_zyx, labels_out = np.concatenate(points), np.concatenate(values)
    if (points_zyx < 0).any() or (points_zyx >= np.asarray(shape_zyx)).any():
        raise ValueError(f"{project.name}'s labels lie outside {raw_image_path.name}.")
    return points_zyx, labels_out


def _normalised(name: str) -> str:
    text = re.sub(r"[_\s]+", " ", name.strip().lower())
    text = _SYNONYMS.get(text, text)
    if text.endswith("ies"):
        text = text[:-3] + "y"
    elif text.endswith("s") and not text.endswith("ss"):
        text = text[:-1]
    return _SYNONYMS.get(text, text)


def default_label_mapping(
    names: Sequence[str], labels: Sequence[str], *, vessel_label: str | int | None = None
) -> dict[int, str | None]:
    """Which of *labels* (a classifier's, the background first) each ilastik
    label (numbered from 1, named *names*) most likely is: the one of the
    same name, else -- for a classifier of the background and one other
    label, given the ilastik label that is vessel (*vessel_label*, a name or
    number, as ``ilastik_vessel_label`` holds it) -- that one for it and the
    background for the rest; None for one left to choose. The vessel label
    is read as the run reads it (``io.ilastik.resolve_vessel_label``)."""
    from haemolynx.io.ilastik import resolve_vessel_label

    by_name = {_normalised(label): label for label in labels}
    mapping: dict[int, str | None] = {
        number: by_name.get(_normalised(name)) for number, name in enumerate(names, start=1)
    }
    if vessel_label not in (None, "") and len(labels) == 2 and None in mapping.values():
        try:
            number = resolve_vessel_label(vessel_label, list(names))
        except ValueError:
            number = None
        if number in mapping:
            for index in mapping:
                if mapping[index] is None:
                    mapping[index] = labels[1] if index == number else labels[0]
    return mapping


def strokes_from_ilastik(
    points: np.ndarray, ilastik_labels: np.ndarray, mapping: Mapping[int, str | None], labels: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """The imported voxels as strokes: ``(points, values)``, each value the
    classifier label's index + 1 (as the training window paints), the
    ilastik labels mapped to None left out."""
    lookup = np.zeros(int(max([0, *mapping.keys(), *ilastik_labels.tolist()])) + 1, np.uint8)
    for number, label in mapping.items():
        if label is not None:
            lookup[int(number)] = labels.index(label) + 1
    values = lookup[ilastik_labels.astype(np.int64)]
    keep = values > 0
    return points[keep], values[keep]
