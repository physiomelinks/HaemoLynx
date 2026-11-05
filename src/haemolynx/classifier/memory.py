"""How much memory a raw image takes to read, and the guards that keep a
read from asking for more than the machine has.

A raw image too large for memory is read lazily (the low-RAM option); read
whole, it is refused before the read rather than left to exhaust memory and
take napari down with it. A lazy reader asked for more at once than the
memory free -- napari's 3D view asks for the whole volume -- raises
``MemoryError`` with what to do, rather than trying.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

#: Reads smaller than this are never weighed against the memory free.
CHECKED_READ_BYTES = 256 << 20

#: The share of the memory free a raw image read whole may take: past it the
#: training window refuses to read it and preflight warns.
OPEN_SHARE = 0.7


def memory_bytes() -> tuple[int | None, int | None]:
    """``(available, total)`` physical memory in bytes, None where unknown
    (without psutil, which napari brings)."""
    try:
        import psutil
    except ImportError:
        return None, None
    memory = psutil.virtual_memory()
    return int(memory.available), int(memory.total)


def gigabytes(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def raw_image_memory_bytes(path: str | Path, channels: Sequence[int | None], axis_order: str = "zyx") -> int:
    """About the memory reading *channels* of the raw TIFF at *path* whole
    takes, as the in-memory read does: only the channels' pages of a file
    stored a page per plane (the whole file otherwise), and a reordered copy
    besides for an axis order other than z, y, x."""
    import tifffile

    from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER, axis_order_for_file, normalize_axis_order
    from haemolynx.io.channels import tiff_channel_axis

    path = Path(path)
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        shape = tuple(int(n) for n in series.shape)
        channel_axis = tiff_channel_axis(axes, shape, tif.imagej_metadata, tif.ome_metadata if tif.is_ome else None)
        total = int(np.prod(shape, dtype=np.int64)) * int(series.dtype.itemsize)
        paged = axes.endswith("YX") and len(series.pages) == int(np.prod(shape[:-2], dtype=np.int64))
    needed = total // shape[channel_axis] * max(1, len(channels)) if channel_axis is not None and paged else total
    try:
        order = normalize_axis_order(axis_order_for_file(axis_order, path))
    except (ValueError, OSError):
        order = CANONICAL_AXIS_ORDER
    return needed * 2 if order != CANONICAL_AXIS_ORDER else needed


def refuse_if_too_large(nbytes: int, what: str) -> None:
    """Raise ``MemoryError`` when reading *nbytes* (*what*) at once would
    take more than the memory free; reads under :data:`CHECKED_READ_BYTES`,
    or where the memory free is unknown, are let through."""
    if nbytes < CHECKED_READ_BYTES:
        return
    available, _total = memory_bytes()
    if available is not None and nbytes > available:
        raise MemoryError(
            f"Reading {what} at once would take {gigabytes(nbytes)}, more than the {gigabytes(available)} "
            "of memory free, so it was not read. napari's 3D view reads a whole image: view a large "
            "image in 2D."
        )
