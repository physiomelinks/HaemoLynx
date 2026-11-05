"""A channel of a TIFF read lazily, a plane at a time, for images far larger than memory.

A stack saved a page per plane (ImageJ, OME and most microscopes' TIFFs) is
read here only where it is looked at: indexing a :class:`LazyTiffChannel`
decompresses just the pages of the planes asked for, and keeps the most
recently read in a cache of bounded size, so the next box over reads them from
memory. It is a NumPy-like array for everything the classifier does with a
volume (``shape``, ``dtype``, basic slicing), and for napari to draw a plane of.
Each worker thread reads through a file handle of its own, so threads
decompress side by side.

The voxels it gives are the file's own, so a classifier reads the same values
from it as from the whole channel loaded into memory.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

#: Memory the decompressed pages may take, unless a caller says otherwise.
PAGE_CACHE_BYTES = 2 << 30


class LazyTiffChannel:
    """One channel of the page-per-plane TIFF at *path*, axes ``(z, y, x)``
    as its pages run (``channel`` None for a file of one channel)."""

    def __init__(self, path: str | Path, channel: int | None = None, *, cache_bytes: int = PAGE_CACHE_BYTES):
        import tifffile

        from haemolynx.io.channels import tiff_channel_axis

        self.path = Path(path)
        self.channel = channel
        index = 0 if channel is None else int(channel)
        with tifffile.TiffFile(str(self.path)) as tif:
            series = tif.series[0]
            axes = series.axes.upper()
            shape = tuple(int(n) for n in series.shape)
            channel_axis = tiff_channel_axis(
                axes, shape, tif.imagej_metadata, tif.ome_metadata if tif.is_ome else None
            )
            leading = list(shape[:-2])
            if not axes.endswith("YX") or len(series.pages) != int(np.prod(leading, dtype=np.int64)):
                raise ValueError(f"{self.path.name} is not stored a page per plane; it cannot be read lazily.")
            if channel_axis is not None and channel_axis >= len(leading):
                raise ValueError(f"{self.path.name} keeps its channels within each page; it cannot be read lazily.")
            if channel_axis is None and index != 0:
                raise ValueError(f"{self.path.name} has no channel axis (axes {axes}); channel {index} does not exist.")
            if channel_axis is not None and not 0 <= index < shape[channel_axis]:
                raise ValueError(
                    f"{self.path.name} has {shape[channel_axis]} channels (0-{shape[channel_axis] - 1}); "
                    f"channel {index} does not exist."
                )
            if leading:
                grid = np.indices(leading).reshape(len(leading), -1)
                pick = np.ones(grid.shape[1], dtype=bool) if channel_axis is None else grid[channel_axis] == index
                self._pages = np.ravel_multi_index(grid[:, pick], leading).astype(np.int64)
            else:
                self._pages = np.zeros(1, dtype=np.int64)
            self.dtype = np.dtype(series.dtype)
            self.shape = (int(len(self._pages)),) + shape[-2:]
        self.ndim = 3
        #: An uncompressed file in one contiguous block is mapped, not read:
        #: a box reads only its own bytes, however large a plane is.
        self._mapped = _mapped_channel(self.path, channel_axis, index, self.shape)
        self._plane_bytes = int(self.shape[1]) * int(self.shape[2]) * self.dtype.itemsize
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._local = threading.local()
        self._handles: list = []

    # A worker process opens the file itself: only where to read is sent.
    def __getstate__(self):
        return {"path": str(self.path), "channel": self.channel, "cache_bytes": self.cache_bytes}

    def __setstate__(self, state):
        self.__init__(state["path"], state["channel"], cache_bytes=state["cache_bytes"])

    # ------------------------------------------------------------- numpy-like

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> np.ndarray:
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            at = key.index(Ellipsis)
            key = key[:at] + (slice(None),) * (3 - len(key) + 1) + key[at + 1:]
        key = key + (slice(None),) * (3 - len(key))
        if len(key) != 3:
            raise IndexError(f"too many indices for a 3D volume: {key!r}")
        if self._mapped is not None:
            from .memory import refuse_if_too_large

            region = self._mapped[key]
            refuse_if_too_large(int(np.size(region)) * self.dtype.itemsize, f"part of {self.path.name}")
            return np.array(region)
        z, rest = key[0], key[1:]
        for part in rest:
            if not isinstance(part, (slice, int, np.integer)):
                raise IndexError("a lazily read channel takes slices and integers in y and x")
        if isinstance(z, (int, np.integer)):
            z = int(z) + (self.shape[0] if z < 0 else 0)
            if not 0 <= z < self.shape[0]:
                raise IndexError(f"plane {key[0]} is outside 0-{self.shape[0] - 1}")
            return self.plane(z)[rest]
        if isinstance(z, slice):
            planes = range(*z.indices(self.shape[0]))
        else:
            planes = [int(p) + (self.shape[0] if p < 0 else 0) for p in np.asarray(z).ravel()]
        probe = np.empty(self.shape[1:], dtype=bool)[rest]
        from .memory import refuse_if_too_large

        refuse_if_too_large(
            len(planes) * probe.size * self.dtype.itemsize, f"{len(planes)} planes of {self.path.name}"
        )
        out = np.empty((len(planes),) + probe.shape, dtype=self.dtype)
        for row, plane in enumerate(planes):
            out[row] = self.plane(plane)[rest]
        return out

    def __array__(self, dtype=None, copy=None):
        whole = self[:]
        return whole if dtype is None else whole.astype(dtype)

    # ------------------------------------------------------------- reading

    def _handle(self):
        import tifffile

        handle = getattr(self._local, "tif", None)
        if handle is None:
            handle = tifffile.TiffFile(str(self.path))
            self._local.tif = handle
            self._handles.append(handle)
        return handle

    def plane(self, index: int) -> np.ndarray:
        """Plane *index* (read-only), from the cache or the file."""
        index = int(index)
        if self._mapped is not None:
            return np.asarray(self._mapped[index])
        with self._cache_lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached
        plane = self._handle().pages[int(self._pages[index])].asarray()
        plane = np.asarray(plane).reshape(self.shape[1:])
        plane.setflags(write=False)
        with self._cache_lock:
            self._cache[index] = plane
            self._cache.move_to_end(index)
            while len(self._cache) > 1 and len(self._cache) * self._plane_bytes > self.cache_bytes:
                self._cache.popitem(last=False)
        return plane

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                pass
        self._handles.clear()
        with self._cache_lock:
            self._cache.clear()

    def __del__(self):
        self.close()


def _mapped_channel(path: Path, channel_axis: int | None, channel: int, shape: tuple[int, int, int]):
    """A read-only memory map of the channel, ``(z, y, x)``, when the file
    holds its voxels uncompressed in one block; else None."""
    import tifffile

    try:
        mapped = tifffile.memmap(str(path), mode="r")
    except (ValueError, OSError, TypeError):
        return None
    if channel_axis is not None:
        if mapped.ndim <= channel_axis + 2:
            return None
        mapped = mapped[(slice(None),) * channel_axis + (channel,)]
    leading = mapped.shape[:-2]
    if len([n for n in leading if n != 1]) > 1:
        return None  # more than one axis of planes: read a page at a time
    # Indexing, never reshaping: a reshape of a strided map can copy it whole.
    mapped = mapped[tuple(0 if n == 1 else slice(None) for n in leading)]
    if mapped.ndim == 2:
        mapped = mapped[np.newaxis]
    return mapped if tuple(mapped.shape) == tuple(shape) else None


def can_read_lazily(path: str | Path, channel: int | None = None) -> bool:
    """Whether :class:`LazyTiffChannel` can read *channel* of *path*."""
    try:
        LazyTiffChannel(path, channel, cache_bytes=0).close()
    except (ValueError, OSError, IndexError):
        return False
    return True


class BlockMeanVolume:
    """*volume* read on a grid *factors* coarser along z, y and x: each voxel
    the mean of its block of the image's voxels (fewer at the far faces), as
    float32, computed only where it is read. What the classifier reads when
    asked to classify on a coarser grid (``pixel_classifier_downsample``)."""

    def __init__(self, volume, factors):
        self.volume = volume
        self.factors = tuple(int(f) for f in factors)
        self.shape = tuple(-(-int(n) // f) for n, f in zip(volume.shape, self.factors))
        self.dtype = np.dtype(np.float32)
        self.ndim = 3

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key) -> np.ndarray:
        from .features import _block_mean

        key = key if isinstance(key, tuple) else (key,)
        key = key + (slice(None),) * (3 - len(key))
        wanted, squeeze, fine = [], [], []
        for part, n, f, total in zip(key, self.shape, self.factors, self.volume.shape):
            if isinstance(part, (int, np.integer)):
                index = np.array([int(part) + (n if part < 0 else 0)])
                squeeze.append(True)
            elif isinstance(part, slice):
                index = np.arange(*part.indices(n))
                squeeze.append(False)
            else:
                raise IndexError("a block-mean view takes slices and integers")
            low, high = (int(index.min()), int(index.max()) + 1) if len(index) else (0, 0)
            wanted.append(index - low)
            fine.append(slice(low * f, min(int(total), high * f)))
        coarse = _block_mean(np.asarray(self.volume[tuple(fine)]), self.factors)
        out = coarse[np.ix_(*wanted)]
        return out.reshape([len(w) for w, s in zip(wanted, squeeze) if not s])

    def close(self) -> None:
        close = getattr(self.volume, "close", None)
        if close is not None:
            close()
