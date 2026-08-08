"""Running the pipeline on the image already open in napari.

The pipeline reads its input from disk: `segment()` takes `input_path` and
`skeletonise()` loads it. A napari layer is an array in memory that may or may
not have come from a file, so the job here is to work out what a given layer
means for the settings -- which file to read, and at what voxel size.

Nothing here imports napari. A layer is used through the three attributes that
matter (`data`, `scale`, `source.path`), so the decisions are testable against
a stand-in, and the widget only has to do the file writing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Suffixes `io.load_and_skeletonize_3d_*` can read back.
READABLE_SUFFIXES = {".tif", ".tiff", ".h5"}


@dataclass(frozen=True)
class LayerInput:
    """What a layer means for the settings, and what had to be done to get it."""

    #: Settings to apply, e.g. `input_path` and the voxel size.
    settings: dict[str, Any]
    #: One line for the panel's report box.
    note: str
    #: True when the layer's array has to be written out before a run.
    needs_export: bool = False


def rejection_reason(layer: Any) -> str | None:
    """Why this layer cannot be the pipeline's input, or None if it can be.

    The pipeline skeletonises a 3D volume, so a 2D image or an RGB stack is a
    mistake worth naming rather than a crash three stages later.
    """
    data = getattr(layer, "data", None)
    if data is None:
        return "That layer has no image data."
    shape = getattr(data, "shape", ())
    if len(shape) != 3:
        return (
            f"The pipeline needs a 3D volume; this layer is {len(shape)}D "
            f"with shape {tuple(shape)}."
        )
    if getattr(layer, "rgb", False):
        return "That layer is RGB; segment it to a single channel first."
    return None


def source_path_of(layer: Any) -> Path | None:
    """The file the layer was read from, if it is one the loaders can re-read.

    napari records where a layer came from. When that is a TIFF or HDF5 on
    disk, the run should read that file rather than a copy of the array: it is
    the same bytes, and the voxel size in its metadata is preserved.
    """
    source = getattr(layer, "source", None)
    path = getattr(source, "path", None) if source is not None else None
    if not path:
        return None
    resolved = Path(path)
    if resolved.suffix.lower() not in READABLE_SUFFIXES:
        return None
    return resolved if resolved.is_file() else None


def voxel_size_xyz_from_scale(scale: Any) -> tuple[float, float, float] | None:
    """A layer's `scale` as the image-metadata voxel size, or None if trivial.

    napari scales a 3D layer per array axis, canonical **(z, y, x)**; the
    setting is image metadata order, **(x, y, z)**. They are reverses of each
    other, and mixing them swaps the z and x spacings -- invisible on isotropic
    data and wrong on every real stack.

    A scale of all ones carries no information, so it is left alone rather than
    overriding whatever the file says.
    """
    if scale is None:
        return None
    values = [float(v) for v in scale]
    if len(values) != 3:
        return None
    if all(value == 1.0 for value in values):
        return None
    z, y, x = values
    return (x, y, z)


def export_name_for(layer: Any) -> str:
    """A filename for a layer that has to be written out before a run."""
    import re

    name = str(getattr(layer, "name", "") or "layer")
    # Runs of anything unsafe collapse to one underscore, so "nerve [1]" is
    # nerve_1 rather than nerve__1.
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return f"{safe or 'layer'}.tif"


def input_for_layer(layer: Any, export_dir: Path | None = None) -> LayerInput:
    """The settings that make a run read *layer*, and what that required.

    A layer loaded from a TIFF or HDF5 points the run at that file. One built
    in the viewer -- a threshold result, a crop -- has no file behind it, so it
    is written to *export_dir* first, and `needs_export` says so.
    """
    reason = rejection_reason(layer)
    if reason is not None:
        raise ValueError(reason)

    settings: dict[str, Any] = {}
    voxel_size = voxel_size_xyz_from_scale(getattr(layer, "scale", None))
    if voxel_size is not None:
        settings["voxel_size_override_xyz"] = list(voxel_size)
        settings["voxel_size_policy"] = "override"

    path = source_path_of(layer)
    if path is not None:
        settings["input_path"] = path
        note = f"Reading {path.name}, the file this layer came from."
        if voxel_size is not None:
            note += f" Voxel size from the layer's scale: {voxel_size} (x, y, z)."
        return LayerInput(settings=settings, note=note, needs_export=False)

    target = Path(export_dir or Path.cwd()) / export_name_for(layer)
    settings["input_path"] = target
    note = (
        f"This layer has no file behind it, so its array will be written to "
        f"{target} and read back."
    )
    if voxel_size is not None:
        note += f" Voxel size from the layer's scale: {voxel_size} (x, y, z)."
    return LayerInput(settings=settings, note=note, needs_export=True)
