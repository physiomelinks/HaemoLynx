"""Masks kept on disk a slab at a time while a segmentation makes them, so
one stopped part-way picks up where it stopped.

Each mask's finished slabs go into a folder beside its output
(``{output}.slabs``), one compressed TIFF per slab named by the planes it
holds, with ``record.json`` saying how they were made: the same record that
decides whether a finished output is current (``apply.classifier_provenance``).
A later segmentation making that mask the same way keeps them, and classifies
only the planes after the last slab every mask of its pass has; one made
otherwise (another classifier, image, channel, voxel size, axis order or grid)
starts afresh. The output is the slabs put together, so a mask made in one go,
or stopped and taken up any number of times with any slab size, is the same
bit for bit: every voxel is classified the same however the image is cut.

Slabs are numbered in the grid's planes (``pixel_classifier_downsample``) and
hold the planes at the image's own size.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

__all__ = ["SLAB_FOLDER_SUFFIX", "SlabStore", "full_size_planes", "slab_folder", "slab_paths"]

#: The folder beside an output its slabs are kept in.
SLAB_FOLDER_SUFFIX = ".slabs"
RECORD_NAME = "record.json"
_SLAB_SUFFIX = ".tif"


def slab_folder(output: str | Path) -> Path:
    output = Path(output)
    return output.with_name(output.name + SLAB_FOLDER_SUFFIX)


def full_size_planes(
    z0: int, planes: np.ndarray, factors: Sequence[int], shape_zyx: Sequence[int]
) -> Iterator[tuple[int, np.ndarray]]:
    """``(plane, voxels)`` at the image's own size for grid planes *planes*
    starting at grid plane *z0*: each grid voxel a block of *factors* voxels,
    cut at the image's edges."""
    fz, fy, fx = (int(f) for f in factors)
    depth, rows, columns = (int(n) for n in shape_zyx)
    for k in range(planes.shape[0]):
        plane = planes[k]
        if fy > 1 or fx > 1:
            plane = np.repeat(np.repeat(plane, fy, axis=0), fx, axis=1)[:rows, :columns]
        for z in range((z0 + k) * fz, min(depth, (z0 + k + 1) * fz)):
            yield z, plane


def _slab_name(z0: int, z1: int) -> str:
    return f"{z0:07d}-{z1:07d}{_SLAB_SUFFIX}"


def slab_paths(output: str | Path) -> list[Path]:
    """The files a store for *output* writes, as long as any it writes (a
    slab's ``.part`` file, its record): what an output path check must see."""
    folder = slab_folder(output)
    return [folder / (_slab_name(0, 0) + ".part"), folder / RECORD_NAME]


def _slab_planes(name: str) -> tuple[int, int] | None:
    if not name.endswith(_SLAB_SUFFIX):
        return None
    try:
        z0, z1 = (int(part) for part in name[: -len(_SLAB_SUFFIX)].split("-"))
    except ValueError:
        return None
    return (z0, z1) if 0 <= z0 < z1 else None


class SlabStore:
    """One mask's slabs (see the module docstring), written on a thread of
    its own. *record* is how the mask is made; *keep*, whether slabs a
    stopped segmentation left that were made the same way are kept."""

    def __init__(
        self,
        output: str | Path,
        record: dict,
        *,
        keep: bool,
        factors: Sequence[int] = (1, 1, 1),
        shape_zyx: Sequence[int],
        compression: dict | None = None,
    ):
        self.output = Path(output)
        self.folder = slab_folder(self.output)
        self.factors = tuple(int(f) for f in factors)
        self.shape_zyx = tuple(int(n) for n in shape_zyx)
        self.compression = dict(compression or {})
        self.record = json.loads(json.dumps(record))  # as it reads back
        recorded = None
        try:
            recorded = json.loads((self.folder / RECORD_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        if not (keep and recorded == self.record):
            self.remove()
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / RECORD_NAME).write_text(json.dumps(self.record, indent=2), encoding="utf-8")
        for stray in self.folder.glob("*.part"):
            stray.unlink(missing_ok=True)
        # One slab waiting while one is written: the pass waits rather than
        # holding more slabs than its memory budget planned for.
        self._queue: "queue.Queue[tuple[int, int, np.ndarray] | None]" = queue.Queue(maxsize=1)
        self._error: list[BaseException] = []
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ reading

    def slabs(self) -> list[tuple[int, int]]:
        """The slabs kept, in plane order."""
        found = (_slab_planes(path.name) for path in self.folder.glob(f"*{_SLAB_SUFFIX}"))
        return sorted(planes for planes in found if planes is not None)

    def finished_planes(self) -> int:
        """How many grid planes from the first the kept slabs hold without a gap."""
        end = 0
        for z0, z1 in self.slabs():
            if z0 != end:
                break
            end = z1
        return end

    def keep_before(self, plane: int) -> None:
        """Drop every slab reaching past grid plane *plane*, or after a gap."""
        end = 0
        for z0, z1 in self.slabs():
            if z0 != end or z1 > plane:
                (self.folder / _slab_name(z0, z1)).unlink(missing_ok=True)
            else:
                end = z1

    def planes(self) -> Iterator[np.ndarray]:
        """The mask's planes at the image's own size, slab by slab."""
        import tifffile

        for z0, z1 in self.slabs():
            with tifffile.TiffFile(str(self.folder / _slab_name(z0, z1))) as tif:
                for page in tif.pages:
                    yield page.asarray()

    # ------------------------------------------------------------ writing

    def put(self, z0: int, z1: int, planes: np.ndarray) -> None:
        """Keep grid planes *z0* to *z1* (*planes*, the grid's), written as
        the image's own planes once the writer gets to them."""
        if self._error:
            raise self._error[0]
        if self._thread is None:
            self._thread = threading.Thread(target=self._write, daemon=True)
            self._thread.start()
        self._queue.put((int(z0), int(z1), planes))

    def _write(self) -> None:
        import tifffile

        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._error:
                continue
            z0, z1, planes = item
            try:
                depth, rows, columns = self.shape_zyx
                fz = self.factors[0]
                count = len(range(z0 * fz, min(depth, z1 * fz)))
                target = self.folder / _slab_name(z0, z1)
                partial = target.with_name(target.name + ".part")
                # A plane at a time: at a coarse grid a slab at full size is
                # many times the slab classified.
                # A page per plane, greyscale: three or four planes would
                # otherwise be stored as one colour image.
                tifffile.imwrite(
                    partial,
                    (np.ascontiguousarray(plane, dtype=np.uint8)
                     for _z, plane in full_size_planes(z0, planes, self.factors, self.shape_zyx)),
                    shape=(count, rows, columns), dtype=np.uint8, photometric="minisblack",
                    **self.compression,
                )
                os.replace(partial, target)
            except BaseException as error:  # noqa: BLE001 - raised by the next put or close
                self._error.append(error)

    def close(self) -> None:
        """Wait for every slab handed over to be written; raise what failed."""
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
        if self._error:
            raise self._error[0]

    def remove(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)
