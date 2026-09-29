"""The channels of a multi-channel TIFF, named the way Fiji names them.

A setting that picks one channel of a composite (``endothelial_channel``,
``fwhm_raw_channel``) stores it counting from 0, as numpy does; Fiji calls the
same channels C1, C2, ... So the panel lists them by Fiji's name, with what
the file says about each: its name, where it has one (OME metadata, or an
ImageJ hyperstack's slice labels), else the colour of its display LUT --
enough to tell a red plasma channel from a green endothelial one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["TiffChannel", "tiff_channel_axis", "tiff_channels"]


@dataclass(frozen=True)
class TiffChannel:
    """One channel: its index (counting from 0) and what the file says about it."""

    index: int
    name: str | None = None
    colour: str | None = None

    @property
    def label(self) -> str:
        """``C2 (claudin-5)``, ``C2 (green)`` or ``C2``: Fiji's name, and the file's."""
        detail = self.name or self.colour
        return f"C{self.index + 1} ({detail})" if detail else f"C{self.index + 1}"


_COLOURS = {
    (1, 0, 0): "red", (0, 1, 0): "green", (0, 0, 1): "blue",
    (1, 1, 0): "yellow", (1, 0, 1): "magenta", (0, 1, 1): "cyan", (1, 1, 1): "grey",
}


def _lut_colour(lut) -> str | None:
    """The colour a display LUT ends on, named when it is a primary or a mix of two."""
    table = np.asarray(lut)
    if table.ndim != 2 or table.shape[0] != 3:
        return None
    top = table[:, -1].astype(float)
    if top.max() <= 0:
        return None
    return _COLOURS.get(tuple(int(v > 0.5 * top.max()) for v in top))


def tiff_channel_axis(axes: str, shape, imagej_metadata=None) -> int | None:
    """Which of a TIFF series' axes holds channels, or ``None``.

    A ``C`` axis alone is not enough: tifffile saves a plain ``(z, y, x)``
    array with ``imagej=True`` as a hyperstack of *z* channels (axes ``CYX``),
    and the loaders read that file as the z-stack it is. So a ``C`` axis with
    no other stack axis beside it counts only when the file says it is a
    colour composite -- an ImageJ ``composite``/``color`` mode or display LUTs.
    """
    axes = axes.upper()
    if "C" not in axes:
        return None
    index = axes.index("C")
    stacked = any(
        int(size) > 1 for axis, size in zip(axes, shape) if axis not in "CYXS"
    )
    if stacked:
        return index
    metadata = imagej_metadata or {}
    if str(metadata.get("mode", "")).lower() in ("composite", "color") or metadata.get("LUTs"):
        return index
    return None


def tiff_channels(path: str | Path | None) -> list[TiffChannel]:
    """The channels of the TIFF at *path*; empty when it has no channel axis
    (a single-channel stack) or cannot be read -- never raises."""
    if not path:
        return []
    path = Path(path)
    if not path.is_file():
        return []
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            axes = series.axes.upper()
            metadata = tif.imagej_metadata or {}
            channel_axis = tiff_channel_axis(axes, series.shape, metadata)
            if channel_axis is None:
                return []
            count = int(series.shape[channel_axis])
            names: list[str | None] = [None] * count
            colours: list[str | None] = [None] * count
            if tif.is_ome and tif.ome_metadata:
                import re

                found = re.findall(r"<Channel\b[^>]*\bName=\"([^\"]*)\"", tif.ome_metadata)
                for index, name in enumerate(found[:count]):
                    names[index] = name or None
            labels = metadata.get("Labels") or []
            if labels and not any(names) and channel_axis == 1 and len(labels) >= count:
                # A hyperstack's slice labels run channel-fastest (Z, C, ...),
                # so the first plane's are the channels'.
                names = [str(label) if label else None for label in labels[:count]]
            for index, lut in enumerate((metadata.get("LUTs") or [])[:count]):
                colours[index] = _lut_colour(lut)
    except Exception:  # noqa: BLE001 - listing channels must never break the form
        return []
    return [TiffChannel(index, names[index], colours[index]) for index in range(count)]
