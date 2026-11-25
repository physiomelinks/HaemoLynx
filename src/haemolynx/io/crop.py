"""Crop on import: one box of voxels cut out of every volume a run reads.

The Input tab's "Crop on import" box has a run analyse only part of its image.
The box is given one of two ways (``crop_method``):

* ``"size"`` -- how many voxels along x, y and z (``crop_size_xyz``), placed at
  one of nine positions in the (x, y) plane (``crop_location``: top left ...
  bottom right, as napari draws a slice, y running down) and one of three
  along z (``crop_z_location``: the first slices, the middle, the last);
* ``"coordinates"`` -- the voxel indices of two opposite corners
  (``crop_start_xyz`` to ``crop_end_xyz``, both kept), as napari shows them
  under the cursor.

Sizes and coordinates are written ``(x, y, z)``, the way a user reads them;
arrays are canonical ``(z, y, x)`` (see :mod:`haemolynx.io.axis_order`), and a
:class:`CropBox` holds ``(z, y, x)`` indices like every other index here. Both
are voxel indices of the image *after* ``image_axis_order`` has been applied.

The box is worked out from each volume's own shape, so every volume read
The box is worked out once, from the main image's whole shape. The vessel
masks are cut to that box as they are loaded; the graph records it
(:data:`GRAPH_ATTRIBUTE`), and every later volume read alongside the main image
-- the raw images, the pericyte and cell masks -- is cut by the graph's record
(:func:`crop_volume_like_graph`), after checking it has the whole image's shape.
So every volume stays aligned with the main image voxel for voxel. Everything a
run writes afterwards (node positions, edge voxels, boundary coordinates) is
measured from the box's corner, not the image's: :meth:`CropBox.offset_um` is
what to add to map a position back.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CROP_LOCATIONS",
    "CROP_METHODS",
    "CROP_Z_LOCATIONS",
    "CropBox",
    "GRAPH_ATTRIBUTE",
    "crop_box",
    "crop_box_from_settings",
    "crop_problems",
    "crop_volume",
    "crop_volume_and_box",
    "crop_volume_from_settings",
    "crop_volume_like_graph",
    "graph_crop_box",
]

CROP_METHODS = ("size", "coordinates")

#: Where a box of ``crop_size_xyz`` sits in the (x, y) plane: (row, column)
#: as napari draws a slice, y running down the screen and x across it.
CROP_LOCATIONS = (
    "top_left",
    "top_centre",
    "top_right",
    "centre_left",
    "centre",
    "centre_right",
    "bottom_left",
    "bottom_centre",
    "bottom_right",
)

#: Where it sits along z: the first slices (z = 0), the middle, the last.
CROP_Z_LOCATIONS = ("first", "centre", "last")

#: The graph attribute holding the box a network was built in
#: (:meth:`CropBox.to_record`); absent when the image was not cropped. What
#: every later reader of a volume aligned with the main image cuts it by.
GRAPH_ATTRIBUTE = "crop_on_import"

#: How far along its axis a box sits, 0 at the start, 1 at the end.
_PLACEMENT = {"top": 0.0, "left": 0.0, "first": 0.0, "centre": 0.5, "bottom": 1.0, "right": 1.0, "last": 1.0}

_XYZ_TO_ZYX = (2, 1, 0)


@dataclass(frozen=True)
class CropBox:
    """Canonical ``(z, y, x)`` voxel indices: *start* kept, *stop* not (Python's
    slice convention), cut from a volume of *source_shape_zyx*."""

    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]
    source_shape_zyx: tuple[int, int, int]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(a, b) for a, b in zip(self.start_zyx, self.stop_zyx))  # type: ignore[return-value]

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(b - a for a, b in zip(self.start_zyx, self.stop_zyx))  # type: ignore[return-value]

    @property
    def is_whole_volume(self) -> bool:
        return self.start_zyx == (0, 0, 0) and self.stop_zyx == tuple(self.source_shape_zyx)

    def offset_um(self, voxel_size_zyx: Sequence[float]) -> tuple[float, float, float]:
        """The box's corner in the whole image, physical ``(z, y, x)`` microns:
        what to add to a position the run wrote to place it on the whole image."""
        return tuple(float(a) * float(s) for a, s in zip(self.start_zyx, voxel_size_zyx))  # type: ignore[return-value]

    def to_record(self, voxel_size_zyx: Sequence[float]) -> dict[str, tuple]:
        """What a graph keeps of the box (:data:`GRAPH_ATTRIBUTE`): plain tuples,
        so it pickles and saves with a run."""
        return {
            "start_zyx": tuple(int(v) for v in self.start_zyx),
            "stop_zyx": tuple(int(v) for v in self.stop_zyx),
            "source_shape_zyx": tuple(int(v) for v in self.source_shape_zyx),
            "offset_um_zyx": self.offset_um(voxel_size_zyx),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CropBox":
        return cls(
            tuple(int(v) for v in record["start_zyx"]),  # type: ignore[arg-type]
            tuple(int(v) for v in record["stop_zyx"]),  # type: ignore[arg-type]
            tuple(int(v) for v in record["source_shape_zyx"]),  # type: ignore[arg-type]
        )

    def describe(self) -> str:
        """The box as a user wrote it: inclusive (x, y, z) voxel ranges."""
        ranges = ", ".join(
            f"{axis} {self.start_zyx[i]}-{self.stop_zyx[i] - 1}"
            for axis, i in (("x", 2), ("y", 1), ("z", 0))
        )
        size = " x ".join(str(self.shape_zyx[i]) for i in _XYZ_TO_ZYX)
        whole = " x ".join(str(self.source_shape_zyx[i]) for i in _XYZ_TO_ZYX)
        return f"{ranges} ({size} voxels, x by y by z, of {whole})"


def _three_ints(name: str, value: Any) -> tuple[int, int, int]:
    if value is None:
        raise ValueError(f"{name} is not set: give three whole numbers, x, y and z.")
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} needs three values, x, y and z; got {len(values)}: {values!r}.")
    if any(float(v) != int(v) for v in values):
        raise ValueError(f"{name} counts voxels, so takes whole numbers; got {values!r}.")
    return tuple(int(v) for v in values)  # type: ignore[return-value]


def _placement(location: str) -> tuple[float, float]:
    """``(y, x)`` placement of an (x, y) location name."""
    if location not in CROP_LOCATIONS:
        raise ValueError(f"crop_location must be one of {', '.join(CROP_LOCATIONS)}; got {location!r}.")
    if location == "centre":
        return 0.5, 0.5
    row, column = location.split("_")
    return _PLACEMENT[row], _PLACEMENT[column]


def crop_box(
    shape_zyx: Sequence[int],
    *,
    method: str,
    size_xyz: Sequence[int] | None = None,
    location: str = "centre",
    z_location: str = "centre",
    start_xyz: Sequence[int] | None = None,
    end_xyz: Sequence[int] | None = None,
) -> CropBox:
    """The box *method* describes, in a volume of *shape_zyx*.

    A size larger than the volume along an axis keeps that whole axis, and
    corners outside it are moved onto its edge; both say so in the log. A box
    with nothing left in it, a size of zero or less, a negative corner, or a
    start past its end raises ``ValueError``.
    """
    shape = tuple(int(n) for n in shape_zyx)
    if len(shape) != 3:
        raise ValueError(f"Crop on import needs a 3D volume; got shape {shape}.")
    if method == "size":
        size_zyx = tuple(reversed(_three_ints("crop_size_xyz", size_xyz)))
        if any(s <= 0 for s in size_zyx):
            raise ValueError(f"crop_size_xyz must be at least 1 voxel on every axis; got {tuple(size_xyz)}.")
        if z_location not in CROP_Z_LOCATIONS:
            raise ValueError(
                f"crop_z_location must be one of {', '.join(CROP_Z_LOCATIONS)}; got {z_location!r}."
            )
        y_place, x_place = _placement(location)
        placements = (_PLACEMENT[z_location], y_place, x_place)
        kept = tuple(min(s, n) for s, n in zip(size_zyx, shape))
        for axis, i in (("x", 2), ("y", 1), ("z", 0)):
            if size_zyx[i] > shape[i]:
                logger.warning(
                    "crop_size_xyz asks for %d voxels along %s, but the image has %d; "
                    "the whole %s axis is kept.", size_zyx[i], axis, shape[i], axis,
                )
        # Centred with an odd voxel to spare, the box sits half a voxel early.
        start = tuple(int((n - k) * p) for n, k, p in zip(shape, kept, placements))
        stop = tuple(a + k for a, k in zip(start, kept))
        return CropBox(start, stop, shape)  # type: ignore[arg-type]
    if method == "coordinates":
        first = tuple(reversed(_three_ints("crop_start_xyz", start_xyz)))
        last = tuple(reversed(_three_ints("crop_end_xyz", end_xyz)))
        if any(v < 0 for v in first + last):
            raise ValueError(
                f"Crop corners are voxel indices, 0 or more; got {tuple(start_xyz)} to {tuple(end_xyz)}."
            )
        for axis, i in (("x", 2), ("y", 1), ("z", 0)):
            if first[i] > last[i]:
                raise ValueError(
                    f"crop_start_xyz's {axis} ({first[i]}) is past crop_end_xyz's ({last[i]}): "
                    "the start corner must have the smaller index on every axis."
                )
            if first[i] >= shape[i]:
                raise ValueError(
                    f"crop_start_xyz's {axis} ({first[i]}) is outside the image, whose {axis} "
                    f"runs 0-{shape[i] - 1}: nothing would be left."
                )
            if last[i] >= shape[i]:
                logger.warning(
                    "crop_end_xyz's %s (%d) is past the image's last voxel (%d); the crop "
                    "stops there.", axis, last[i], shape[i] - 1,
                )
        stop = tuple(min(v + 1, n) for v, n in zip(last, shape))
        return CropBox(first, stop, shape)  # type: ignore[arg-type]
    raise ValueError(f"crop_method must be one of {', '.join(CROP_METHODS)}; got {method!r}.")


def crop_box_from_settings(settings: Mapping[str, Any], shape_zyx: Sequence[int]) -> CropBox | None:
    """The box the run's settings ask for, or None while ``crop_on_import`` is off."""
    if not settings.get("crop_on_import"):
        return None
    return crop_box(
        shape_zyx,
        method=settings.get("crop_method", "size"),
        size_xyz=settings.get("crop_size_xyz"),
        location=settings.get("crop_location", "centre"),
        z_location=settings.get("crop_z_location", "centre"),
        start_xyz=settings.get("crop_start_xyz"),
        end_xyz=settings.get("crop_end_xyz"),
    )


def crop_problems(settings: Mapping[str, Any]) -> list[str]:
    """What is wrong with the crop settings before any image is read: what
    :func:`crop_box` would refuse whatever the image's size. Empty when
    ``crop_on_import`` is off."""
    if not settings.get("crop_on_import"):
        return []
    # A volume larger than any real one: only what fails regardless of the
    # image's size can fail here.
    huge = (np.iinfo(np.int32).max,) * 3
    try:
        crop_box_from_settings(settings, huge)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def crop_volume(
    volume: np.ndarray,
    box: CropBox | None,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    binarise=None,
) -> np.ndarray:
    """*volume* cut to *box*, as an array of its own.

    A copy rather than a view, so the whole volume it was cut from is freed
    once the caller lets go of it. With *use_memmap* the copy is a new
    disk-backed array, filled a slab at a time. *volume* itself comes back
    when *box* is None or covers all of it.

    *binarise*, an elementwise rule decided on the whole of *volume*
    (:func:`haemolynx.io.load.mask_binarization_rule`), makes the copy the
    box's boolean mask by that rule, a slab at a time: binarised after the
    cut, the box alone decided what was vessel (an ilastik 1/2 box holding
    only background came back solid vessel).
    """
    if box is None:
        return volume
    if tuple(volume.shape) != tuple(box.source_shape_zyx):
        raise ValueError(
            f"Cannot crop a volume of shape {tuple(volume.shape)} to a box worked out for "
            f"shape {tuple(box.source_shape_zyx)}."
        )
    if box.is_whole_volume:
        return volume
    region = volume[box.slices]
    if binarise is None and not use_memmap:
        return np.array(region, copy=True)
    from haemolynx.preprocessing.memmap_support import new_memmap_array, slab_step

    dtype = volume.dtype if binarise is None else np.dtype(bool)
    if use_memmap:
        out = new_memmap_array(box.shape_zyx, dtype, directory=memmap_directory)
    else:
        out = np.empty(box.shape_zyx, dtype=dtype)
    step = slab_step(box.shape_zyx)
    for start in range(0, box.shape_zyx[0], step):
        slab = region[start:start + step]
        out[start:start + step] = slab if binarise is None else binarise(np.asarray(slab))
    return out


def graph_crop_box(G: Any) -> CropBox | None:
    """The box the network *G* was built in, from its :data:`GRAPH_ATTRIBUTE`;
    None for a network built on the whole image."""
    record = getattr(G, "graph", {}).get(GRAPH_ATTRIBUTE) if G is not None else None
    return None if not record else CropBox.from_record(record)


def crop_volume_like_graph(
    volume: np.ndarray,
    G: Any,
    *,
    description: str,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """*volume* -- read from disk alongside the main image -- cut to the box
    the network *G* was built in, so its voxels land where the network's do.

    *volume* itself when *G* was built on the whole image. A volume whose
    shape is not the whole image's the box was cut from cannot be placed on
    the network at all, and raises ``ValueError`` saying so.
    """
    box = graph_crop_box(G)
    if box is None:
        return volume
    if tuple(volume.shape) != tuple(box.source_shape_zyx):
        raise ValueError(
            f"The {description} is {tuple(volume.shape)} voxels, but the network was built "
            f"in a crop_on_import box cut from an image of {tuple(box.source_shape_zyx)}: "
            f"the {description} must have the input image's whole shape to be cut to the "
            "same box."
        )
    logger.info("Crop on import, %s: %s", description, box.describe())
    return crop_volume(volume, box, use_memmap=use_memmap, memmap_directory=memmap_directory)


def crop_volume_and_box(
    volume: np.ndarray,
    settings: Mapping[str, Any],
    *,
    description: str = "image",
    binarise=None,
) -> tuple[np.ndarray, CropBox | None]:
    """:func:`crop_volume_from_settings`, and the box it cut (None while
    ``crop_on_import`` is off). *binarise* is :func:`crop_volume`'s: a rule
    decided on the whole of *volume*, applied only when a box is cut."""
    box = crop_box_from_settings(settings, volume.shape)
    if box is None or box.is_whole_volume:
        return volume, box
    logger.info("Crop on import, %s: %s", description, box.describe())
    cropped = crop_volume(
        volume,
        box,
        use_memmap=bool(settings.get("use_memmap_loading")),
        memmap_directory=settings.get("memmap_directory"),
        binarise=binarise,
    )
    return cropped, box


def crop_volume_from_settings(
    volume: np.ndarray,
    settings: Mapping[str, Any],
    *,
    description: str = "image",
) -> np.ndarray:
    """*volume* cut to the box the run's settings ask for, worked out from its
    own shape; *volume* itself while ``crop_on_import`` is off. *description*
    names the volume in the log line saying what was kept."""
    return crop_volume_and_box(volume, settings, description=description)[0]
