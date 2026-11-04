"""Segmenting a raw image with saved pixel classifiers, in one pass however many.

The raw image is read channel by channel in canonical ``(z, y, x)`` order, as
every other reader of it reads it -- under the low-RAM option lazily
(:class:`~haemolynx.classifier.lazy.LazyTiffChannel`: an uncompressed file
mapped, a compressed one decoded a page at a time), so an image far larger
than memory is never held whole. It is classified a slab of planes at a time
(:func:`classify_slabs`), each slab in boxes on worker threads: every
classifier asked for votes on one box's features, computed once for all of
them (the union of their channels and scales, and only the features some
tree reads). Each mask asked for is written as a compressed 0/1 TIFF a plane
at a time as its slab finishes -- in the raw file's own axis order and with
its voxel size, so the run reads it back under the same ``image_axis_order``
as the raw image -- with a record beside it (``{output}.haemolynx.json``) of
how it was made. An output whose record matches what this run would make is
used again rather than segmented again: the classifier by its bytes, the raw
image by path, size, time and a sample of its bytes, the channels, the mask,
the voxel size and the axis order.

A voxel's label is the same however the image is cut into slabs and boxes,
however many workers run, whatever else is classified beside it, in RAM or
lazily: its features are the whole volume's, and its forest walk reads only
them.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .features import (
    BOX_VOXELS,
    FeatureSpec,
    coarse_context,
    coarse_reach,
    feature_halo,
    features_of_box,
    has_coarse_scales,
    intensity_range,
)
from .model import PixelClassifier, load_classifier

logger = logging.getLogger(__name__)

#: What a segmentation's record says made it.
SOURCE = "HaemoLynx pixel classifier"

#: The record's layout; a record of any other is not trusted.
PROVENANCE_VERSION = 1

#: The fields that must match this run's for an output to be used again.
_PROVENANCE_KEYS = (
    "version", "source", "classifier_sha256", "input", "label", "voxel_size_zyx", "axis_order",
)

#: About how much memory one box being worked on takes, per padded voxel and
#: per feature of a core voxel (bytes): the scaled channels and a scale's
#: smoothed image and derivatives, then the features and the forest's walk.
_BYTES_PER_PADDED_VOXEL = 64
_BYTES_PER_CORE_FEATURE = 12

#: Never more workers than this, however much memory there is: past it the
#: threads wait on one another.
MAX_WORKERS = 8

#: Voxels per padded box under the low-RAM option. Smaller boxes would hold
#: less, but recompute more of their margin (at 8 um and 0.5 um voxels, 42 a
#: side); the boxes in flight are bounded by the free memory anyway
#: (:func:`worker_count`). The answer is the same whatever the box size.
LOW_RAM_BOX_VOXELS = 4_000_000

#: Planes a slab holds at most, and the memory a low-RAM pass keeps its slabs
#: and decoded pages within (less when less is free).
MAX_SLAB_PLANES = 64

#: The side (voxels) of the tiles of a slab whose coarse scales are computed
#: once and shared by the boxes inside: wide, so their coarse margin is a
#: small part of them.
COARSE_TILE = 1024
LOW_RAM_BUDGET_BYTES = 4 << 30

#: How the masks are compressed: fast, and a 0/1 mask shrinks some fifty-fold.
MASK_COMPRESSION = {"compression": "zlib", "compressionargs": {"level": 1}}


@dataclass(frozen=True)
class RawChannels:
    """The channels of a raw image a classifier reads, as it reads them."""

    path: Path
    volumes: tuple  # canonical (z, y, x): arrays, memory maps or lazy channels
    voxel_size_zyx: tuple[float, float, float]
    axis_order: str  # the order the file was read in
    file_spacing: tuple[float, float, float]  # along the file's own axes
    channels: tuple[int | None, ...] = (None,)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.volumes[0].shape)  # type: ignore[return-value]

    def volume(self, channel: int | None):
        return self.volumes[self.channels.index(channel)]


def read_raw_channels(
    path: str | Path,
    channels: Sequence[int | None],
    *,
    axis_order: str = "zyx",
    voxel_size_policy: str = "auto",
    voxel_size_override_xyz=None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    lazy: bool | None = None,
) -> RawChannels:
    """*channels* of the raw TIFF at *path* (``None`` for a file of one
    channel), read as the run reads its raw image, with the voxel size the
    run's voxel-size policy gives it. *lazy* (the low-RAM option's default)
    reads them where they are looked at (:func:`load_channel`)."""
    import tifffile

    from haemolynx.io.axis_order import (
        axis_order_for_file,
        file_axis_spacing_from_xyz,
        voxel_size_zyx_from_xyz,
    )
    from haemolynx.io.load import _voxel_size_xyz_from_tiff
    from haemolynx.io.voxel_validation import resolve_voxel_size_xyz

    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"The classifier reads TIFF raw images; {path.name} is not one.")
    order = axis_order_for_file(axis_order, path)
    with tifffile.TiffFile(str(path)) as tif:
        x, y, z, status = _voxel_size_xyz_from_tiff(tif, order)
    voxel_xyz, _source = resolve_voxel_size_xyz((x, y, z), status, voxel_size_override_xyz, voxel_size_policy)
    volumes = tuple(
        load_channel(
            path, channel, order, use_memmap=use_memmap, memmap_directory=memmap_directory,
            lazy=use_memmap if lazy is None else lazy,
        )
        for channel in channels
    )
    shapes = {tuple(volume.shape) for volume in volumes}
    if len(shapes) != 1:
        raise ValueError(f"The channels of {path.name} have different shapes: {sorted(shapes)}.")
    return RawChannels(
        path=path,
        volumes=volumes,
        voxel_size_zyx=voxel_size_zyx_from_xyz(voxel_xyz),
        axis_order=order,
        file_spacing=file_axis_spacing_from_xyz(voxel_xyz, order),
        channels=tuple(channels),
    )


#: Pages read at a time into a low-RAM channel.
_PAGES_PER_READ = 16


def load_channel(
    path: str | Path,
    channel: int | None,
    axis_order: str,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    lazy: bool = False,
):
    """One channel of the raw TIFF at *path* (``None``: a file of one
    channel), in canonical order and in the file's own number type -- the
    features scale it to float32 a box at a time, so no float32 copy of the
    image is ever held. A stack of one plane is not reordered, as the run's
    other readers of a raw image do not reorder it.

    *lazy*: read where it is looked at, never whole
    (:class:`~haemolynx.classifier.lazy.LazyTiffChannel`), for a file stored
    a page per plane in canonical order. Otherwise with *use_memmap* (the
    low-RAM option) the channel is copied a few pages at a time into a
    disk-backed array. The same voxels every way."""
    import tifffile

    from haemolynx.haemodynamics.fwhm_diameter import _read_tiff_channel
    from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER, apply_axis_order
    from haemolynx.io.channels import tiff_channel_axis

    path = Path(path)
    index = 0 if channel is None else int(channel)
    if lazy:
        from .lazy import LazyTiffChannel

        try:
            reader = LazyTiffChannel(path, channel)
        except (ValueError, OSError):
            reader = None
        if reader is not None and (axis_order == CANONICAL_AXIS_ORDER or reader.shape[0] == 1):
            return reader
        if reader is not None:
            reader.close()
    if not use_memmap:
        volume = _read_tiff_channel(path, index)
        return volume if volume.shape[0] == 1 else apply_axis_order(volume, axis_order)
    from haemolynx.preprocessing.memmap_support import new_memmap_array, release_memmap_array

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        shape = tuple(int(n) for n in series.shape)
        channel_axis = tiff_channel_axis(
            axes, shape, tif.imagej_metadata, tif.ome_metadata if tif.is_ome else None
        )
        leading = list(shape[:-2])
        paged = axes.endswith("YX") and len(series.pages) == int(np.prod(leading, dtype=np.int64))
        if channel_axis is not None and not 0 <= index < shape[channel_axis]:
            raise ValueError(
                f"{path.name} has {shape[channel_axis]} channels (0-{shape[channel_axis] - 1}); "
                f"channel {index} does not exist."
            )
        if channel_axis is None and index != 0:
            raise ValueError(f"{path.name} has no channel axis (axes {axes}); channel {index} does not exist.")
        if not paged or (channel_axis is not None and channel_axis >= len(leading)):
            volume = _read_tiff_channel(path, index)
        else:
            grid = np.indices(leading).reshape(len(leading), -1) if leading else np.zeros((0, 1), int)
            pick = np.ones(grid.shape[1], dtype=bool)
            if channel_axis is not None:
                pick = grid[channel_axis] == index
            pages = np.ravel_multi_index(grid[:, pick], leading) if leading else np.array([0])
            volume = new_memmap_array((len(pages),) + shape[-2:], series.dtype, directory=memmap_directory)
            for start in range(0, len(pages), _PAGES_PER_READ):
                chunk = pages[start:start + _PAGES_PER_READ]
                for offset, page in enumerate(chunk):
                    volume[start + offset] = series.pages[int(page)].asarray()
            volume.flush()
    if volume.ndim == 2:
        volume = volume[np.newaxis]
    if volume.shape[0] == 1 or axis_order == CANONICAL_AXIS_ORDER:
        return volume
    reordered = apply_axis_order(volume, axis_order, use_memmap=True, memmap_directory=memmap_directory)
    if isinstance(volume, np.memmap):
        release_memmap_array(volume)
    return reordered


def available_memory_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


#: The largest box :func:`box_size` chooses.
MAX_BOX_VOXELS = 32_000_000


def box_size(*, budget_bytes: int | None = None, workers: int | None = None) -> int:
    """Voxels per padded box: as large as a share of *budget_bytes* (half the
    free memory when not given) allows each worker, between
    :data:`BOX_VOXELS` and :data:`MAX_BOX_VOXELS`. A larger box recomputes
    less of its margin; the answer is the same whatever the size."""
    if budget_bytes is None:
        free = available_memory_bytes()
        if free is None:
            return BOX_VOXELS
        budget_bytes = free // 2
    workers = workers or min(os.cpu_count() or 1, MAX_WORKERS)
    per_worker = budget_bytes // max(workers, 1)
    return int(max(BOX_VOXELS, min(MAX_BOX_VOXELS, per_worker // (2 * _BYTES_PER_PADDED_VOXEL))))


def worker_count(box_voxels: int, core_voxels: int, n_features: int, *, workers: int | None = None) -> int:
    """How many boxes to work on at once: *workers* when given, else as many
    as half the free memory holds, at most the cores and :data:`MAX_WORKERS`."""
    if workers is not None:
        return max(1, int(workers))
    cores = os.cpu_count() or 1
    per_box = box_voxels * _BYTES_PER_PADDED_VOXEL + core_voxels * n_features * _BYTES_PER_CORE_FEATURE
    free = available_memory_bytes()
    by_memory = MAX_WORKERS if free is None else max(1, int(0.5 * free // max(per_box, 1)))
    return max(1, min(cores, MAX_WORKERS, by_memory))


# ---------------------------------------------------------- shared features


def shared_spec(specs: Sequence[FeatureSpec]) -> FeatureSpec:
    """The features every classifier of *specs* reads, computed once for all:
    the union of their channels and of their scales."""
    channels = {c for spec in specs for c in spec.channels}
    if None in channels and len(channels) > 1:
        raise ValueError(
            "One classifier reads the whole one-channel raw image and another a channel of a "
            "composite: they cannot both read this raw image."
        )
    grids = {spec.coarse for spec in specs}
    if len(grids) > 1:
        raise ValueError("Classifiers on the coarse grid and on the image's own share no features.")
    scales = sorted({s for spec in specs for s in spec.scales_um})
    ordered = (None,) if channels == {None} else tuple(sorted(channels))
    return FeatureSpec(scales_um=tuple(scales), channels=ordered, coarse=grids.pop())


def spec_columns(spec: FeatureSpec, shared: FeatureSpec) -> np.ndarray:
    """Where each of *spec*'s features sits among *shared*'s."""
    names = shared.names()
    return np.array([names.index(name) for name in spec.names()], dtype=np.int64)


def forest_features(classifier: PixelClassifier) -> np.ndarray:
    """The features (indices into its spec's) some tree of the classifier's
    forest reads."""
    forest = classifier.forest
    branches = forest.left != np.arange(len(forest.left))
    return np.unique(forest.feature[branches])


@dataclass(frozen=True)
class Job:
    """One classifier's share of a pass: each output's key and the labels
    it marks (None: the label itself, for a volume of labels)."""

    classifier: PixelClassifier
    outputs: tuple[tuple[str, tuple[int, ...] | None], ...]


def _slab_core(
    shape: tuple[int, int, int], halo: tuple[int, int, int], slab: tuple[int, int], box_voxels: int
) -> tuple[int, int]:
    """The (y, x) extent of a box's core within a slab, so its padded box
    holds about *box_voxels*, never thinner than twice the margin."""
    z0, z1 = slab
    depth = min(shape[0], z1 + halo[0]) - max(0, z0 - halo[0])
    side = int(np.sqrt(max(box_voxels, 1) / max(depth, 1)))
    cores = []
    for axis in (1, 2):
        core = max(side - 2 * halo[axis], 2 * halo[axis], 16)
        cores.append(min(core, shape[axis]))
    return cores[0], cores[1]


def slab_planes(
    shape: tuple[int, int, int],
    halo: tuple[int, int, int],
    *,
    channels: int,
    itemsize: int,
    outputs: int,
    budget_bytes: int,
) -> int:
    """Planes per slab: as many as *budget_bytes* holds of the slab's own
    outputs and its pages read with their margin, at most
    :data:`MAX_SLAB_PLANES` -- but at least twice the z margin while the
    outputs alone fit, as a thinner slab recomputes its margin for few
    planes (on a large plane, the pages then are read again box by box)."""
    plane = int(shape[1]) * int(shape[2])
    per_plane = plane * (channels * itemsize + outputs)
    fit = budget_bytes // max(per_plane, 1) - 2 * halo[0]
    outputs_fit = budget_bytes // max(plane * max(outputs, 1), 1)
    wanted = min(max(8, 4 * halo[0]), outputs_fit)
    return int(max(1, min(MAX_SLAB_PLANES, max(fit, wanted), shape[0])))


def classify_slabs(
    volumes: Sequence,
    ranges: Sequence[tuple[float, float]],
    voxel_size_zyx: Sequence[float],
    jobs: Sequence[Job],
    *,
    sink: Callable[[int, int, dict[str, np.ndarray]], None],
    workers: int | None = None,
    poll: Optional[Callable[[], None]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    box_voxels: int | None = None,
    planes_per_slab: int | None = None,
) -> None:
    """Classify the image a slab of planes at a time for every job at once,
    handing *sink* each slab's outputs, ``(z0, z1, {key: uint8 (planes, y, x)})``,
    in order. *volumes* and *ranges* are the shared spec's channels
    (:func:`shared_spec` over the jobs'), in its order. *poll* is called
    between boxes: whatever it raises stops the work and propagates.
    *progress* hears ``(boxes done, boxes)``."""
    if not jobs:
        return
    for job in jobs:
        if job.classifier.forest is None:
            raise ValueError("The classifier has not been trained.")
    shared = shared_spec([job.classifier.spec for job in jobs])
    if len(volumes) != len(shared.channels):
        raise ValueError(f"The classifiers read {len(shared.channels)} channel(s); given {len(volumes)}.")
    columns = [spec_columns(job.classifier.spec, shared) for job in jobs]
    needed = np.zeros(shared.n_features, dtype=bool)
    for job, cols in zip(jobs, columns):
        needed[cols[forest_features(job.classifier)]] = True
    shape = tuple(int(n) for n in volumes[0].shape)
    halo = feature_halo(shared, voxel_size_zyx)
    box_voxels = int(box_voxels or BOX_VOXELS)
    if planes_per_slab is None:
        planes_per_slab = slab_planes(
            shape, halo, channels=len(volumes),
            itemsize=int(np.dtype(getattr(volumes[0], "dtype", np.uint16)).itemsize),
            outputs=sum(len(job.outputs) for job in jobs), budget_bytes=LOW_RAM_BUDGET_BYTES,
        )
    slabs = [(z0, min(z0 + planes_per_slab, shape[0])) for z0 in range(0, shape[0], planes_per_slab)]
    # Coarse scales are computed once per tile of the slab, large in y and x,
    # and shared by the boxes inside it (features.coarse_context): no box
    # reads or averages the coarse margin itself.
    tiled = has_coarse_scales(shared, voxel_size_zyx)
    tile = COARSE_TILE if tiled else max(shape[1], shape[2])
    plans = []
    for z0, z1 in slabs:
        by, bx = _slab_core(shape, halo, (z0, z1), box_voxels)
        for ty in range(0, shape[1], tile):
            for tx in range(0, shape[2], tile):
                region = (slice(z0, z1), slice(ty, min(ty + tile, shape[1])), slice(tx, min(tx + tile, shape[2])))
                for y0 in range(ty, region[1].stop, by):
                    for x0 in range(tx, region[2].stop, bx):
                        core = (
                            slice(z0, z1),
                            slice(y0, min(y0 + by, region[1].stop)),
                            slice(x0, min(x0 + bx, region[2].stop)),
                        )
                        plans.append(((z0, z1), region, core))
    largest = max(int(np.prod([s.stop - s.start for s in core])) for _slab, _region, core in plans)
    count = worker_count(box_voxels, largest, int(needed.sum()), workers=workers)
    nf = shared.n_features

    def one(core, context):
        padded = tuple(
            slice(max(0, c.start - h), min(n, c.stop + h)) for c, h, n in zip(core, halo, shape)
        )
        inner = tuple(slice(c.start - p.start, c.stop - p.start) for c, p in zip(core, padded))
        features = features_of_box(
            volumes, ranges, voxel_size_zyx, shared, padded, inner, needed=needed,
            coarse=None if context is None else context.result(),
        )
        flat = features.reshape(-1, nf)
        results = []
        for job, cols in zip(jobs, columns):
            labels = job.classifier.forest.predict(np.ascontiguousarray(flat[:, cols])).reshape(features.shape[:3])
            for key, marks in job.outputs:
                if marks is None:
                    results.append((key, labels))
                elif len(marks) == 1:
                    results.append((key, (labels == marks[0]).astype(np.uint8)))
                else:
                    results.append((key, np.isin(labels, marks).astype(np.uint8)))
        return core, results

    keys = [key for job in jobs for key, _marks in job.outputs]
    done = 0
    position = 0
    with ThreadPoolExecutor(max_workers=count) as pool:
        for z0, z1 in slabs:
            buffers = {key: np.zeros((z1 - z0,) + shape[1:], dtype=np.uint8) for key in keys}
            mine = [(region, core) for slab, region, core in plans[position:] if slab == (z0, z1)]
            position += len(mine)
            pending: set = set()
            queued = iter(mine)
            # Each tile's coarse context is started a few tiles ahead of its
            # boxes, so the boxes rarely wait on one: the pool runs in order,
            # so a box waits only on a context already running.
            regions = list(dict.fromkeys(tuple((r.start, r.stop) for r in region) for region, _core in mine))
            contexts: dict = {}
            ahead = max(2, count // 2)

            def context_for(key):
                at = regions.index(key)
                for upcoming in regions[at:at + ahead]:
                    if upcoming not in contexts:
                        contexts[upcoming] = pool.submit(
                            coarse_context, volumes, ranges, voxel_size_zyx, shared,
                            tuple(slice(a, b) for a, b in upcoming), needed,
                        )
                for passed in [k for k in contexts if regions.index(k) < at]:
                    del contexts[passed]  # its boxes hold their own reference
                return contexts[key]

            try:
                while True:
                    while len(pending) < 2 * count:
                        item = next(queued, None)
                        if item is None:
                            break
                        region, core = item
                        context = context_for(tuple((r.start, r.stop) for r in region)) if tiled else None
                        pending.add(pool.submit(one, core, context))
                    if not pending:
                        break
                    finished, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for future in finished:
                        core, results = future.result()
                        local = (slice(core[0].start - z0, core[0].stop - z0),) + core[1:]
                        for key, values in results:
                            buffers[key][local] = values
                        done += 1
                        if progress is not None:
                            progress(done, len(plans))
                    if poll is not None:
                        poll()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
            sink(z0, z1, buffers)


def classify_volume(
    volumes: Sequence,
    voxel_size_zyx: Sequence[float],
    classifier: PixelClassifier,
    *,
    ranges: Sequence[tuple[float, float]] | None = None,
    out: np.ndarray | None = None,
    workers: int | None = None,
    poll: Optional[Callable[[], None]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    box_voxels: int | None = None,
    planes_per_slab: int | None = None,
) -> np.ndarray:
    """Each voxel's most probable label (uint8, an index into the
    classifier's labels) for the channel *volumes* of one image.

    *ranges* are the channels' intensity ranges (each volume's own by
    default, :func:`~haemolynx.classifier.features.intensity_range`); *out*
    an array to write into (a memory map, for a large image). *poll* is called
    between boxes: whatever it raises stops the work and propagates.
    *progress* hears ``(boxes done, boxes)``.
    """
    if classifier.forest is None:
        raise ValueError("The classifier has not been trained.")
    if len(volumes) != len(classifier.spec.channels):
        raise ValueError(
            f"The classifier reads {len(classifier.spec.channels)} channel(s); given {len(volumes)}."
        )
    if ranges is None:
        ranges = [intensity_range(volume) for volume in volumes]
    shape = tuple(int(n) for n in volumes[0].shape)
    if out is None:
        out = np.zeros(shape, dtype=np.uint8)

    def sink(z0, z1, buffers):
        out[z0:z1] = buffers["labels"]

    classify_slabs(
        volumes, ranges, voxel_size_zyx, [Job(classifier, (("labels", None),))], sink=sink,
        workers=workers, poll=poll, progress=progress, box_voxels=box_voxels, planes_per_slab=planes_per_slab,
    )
    return out


# --------------------------------------------------------------- records


def classifier_provenance(
    raw_path: str | Path,
    classifier_path: str | Path,
    label: str,
    *,
    channels: Sequence[int | None],
    voxel_size_zyx: Sequence[float],
    axis_order: str,
) -> dict:
    """What an output of mask *label* made from *raw_path* by
    *classifier_path* depends on (see the module docstring)."""
    from haemolynx.io.ilastik import _file_identity, file_sha256

    classifier_path = Path(classifier_path)
    record = {
        "version": PROVENANCE_VERSION,
        "source": SOURCE,
        "classifier_sha256": file_sha256(classifier_path),
        "input": _file_identity(Path(raw_path)),
        "label": str(label),
        "voxel_size_zyx": [round(float(v), 9) for v in voxel_size_zyx],
        "axis_order": str(axis_order),
        "classifier": str(classifier_path.resolve()),
    }
    record["input"]["channels"] = [None if c is None else int(c) for c in channels]
    return record


def classifier_output_is_current(output_path: str | Path, provenance: dict) -> tuple[bool, str]:
    """Whether the output at *output_path* was made as *provenance* says
    this run would make it, and why not."""
    from haemolynx.io.ilastik import sidecar_path

    output_path = Path(output_path)
    if not output_path.is_file():
        return False, "there is no output yet"
    sidecar = sidecar_path(output_path)
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, f"it has no {sidecar.name} saying how it was made"
    except (OSError, ValueError):
        return False, f"its record {sidecar.name} cannot be read"
    if not isinstance(recorded, dict):
        return False, f"its record {sidecar.name} cannot be read"
    reasons = {
        "version": "it was recorded by another version of HaemoLynx",
        "source": "it was not made by a HaemoLynx classifier",
        "classifier_sha256": "the classifier is not the one it was made with",
        "input": "the raw image, or the channels read of it, has changed",
        "label": "it is another label's mask",
        "voxel_size_zyx": "the raw image's voxel size has changed",
        "axis_order": "the raw image is read in another axis order",
    }
    for key in _PROVENANCE_KEYS:
        if recorded.get(key) != provenance.get(key):
            return False, reasons[key]
    output = recorded.get("output")
    stat = output_path.stat()
    if not isinstance(output, dict) or (output.get("size"), output.get("mtime_ns")) != (
        stat.st_size,
        stat.st_mtime_ns,
    ):
        return False, "the output file has changed since it was made"
    return True, ""


# --------------------------------------------------------------- writing


def count_label(labels: np.ndarray, index: int) -> int:
    """How many voxels of *labels* are *index*, a plane at a time."""
    return int(sum(np.count_nonzero(np.asarray(labels[z]) == index) for z in range(labels.shape[0])))


def _mask_metadata(raw: RawChannels) -> dict:
    pages, rows, columns = raw.file_spacing
    return dict(
        imagej=True,
        resolution=(1.0 / columns, 1.0 / rows),
        metadata={"axes": "ZYX", "spacing": pages, "unit": "um"},
        software="HaemoLynx pixel classifier",
        **MASK_COMPRESSION,
    )


def _partial(path: Path) -> Path:
    return path.with_name(path.stem + ".part" + path.suffix)


def _streams(raw: RawChannels) -> bool:
    """Whether masks can be written a plane at a time as they are made: the
    raw file's planes are z, as the slabs are."""
    from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER

    return raw.axis_order == CANONICAL_AXIS_ORDER or raw.shape_zyx[0] == 1


class _MaskWriter:
    """One mask written plane by plane, on a thread of its own, into a
    ``.part`` file moved over the name once whole."""

    def __init__(self, path: Path, raw: RawChannels):
        import tifffile

        self.path = path
        self.partial = _partial(path)
        self.vessel = 0
        self._planes: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=8)
        self._error: list[BaseException] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = raw.shape_zyx

        def planes():
            while True:
                plane = self._planes.get()
                if plane is None:
                    return
                yield plane

        def write():
            try:
                tifffile.imwrite(self.partial, planes(), shape=shape, dtype=np.uint8, **_mask_metadata(raw))
            except BaseException as error:  # noqa: BLE001 - raised on close
                self._error.append(error)
                while self._planes.get() is not None:
                    pass

        self._thread = threading.Thread(target=write, daemon=True)
        self._thread.start()

    def put(self, planes: np.ndarray) -> None:
        for plane in planes:
            self.vessel += int(np.count_nonzero(plane))
            self._planes.put(np.ascontiguousarray(plane))

    def close(self) -> None:
        self._planes.put(None)
        self._thread.join()
        if self._error:
            raise self._error[0]
        os.replace(self.partial, self.path)

    def abort(self) -> None:
        self._planes.put(None)
        self._thread.join()
        self.partial.unlink(missing_ok=True)


def _write_mask(path: Path, labels: np.ndarray, raw: RawChannels) -> int:
    """A whole 0/1 mask, canonical order, written in the raw file's own axis
    order (one stored otherwise than z, y, x); returns its vessel voxels."""
    import tifffile

    from haemolynx.io.axis_order import axis_order_transpose

    stored = np.transpose(labels, np.argsort(axis_order_transpose(raw.axis_order)))

    def planes():
        for plane in range(stored.shape[0]):
            yield np.ascontiguousarray(stored[plane], dtype=np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial(path)
    tifffile.imwrite(partial, planes(), shape=tuple(int(n) for n in stored.shape), dtype=np.uint8,
                     **_mask_metadata(raw))
    os.replace(partial, path)
    return count_label(labels, 1)


# ----------------------------------------------------------- segmentation


def segment_with_classifiers(
    raw_image_path: str | Path,
    requests: Sequence[tuple[str | Path, Mapping[str, str | Path]]],
    *,
    axis_order: str = "zyx",
    voxel_size_policy: str = "auto",
    voxel_size_override_xyz=None,
    reuse_existing: bool = True,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    workers: int | None = None,
    poll: Optional[Callable[[], None]] = None,
) -> list[dict[str, Path]]:
    """Segment the raw image with each classifier of *requests* --
    ``(classifier path, {mask: output path})`` -- in one pass, the features
    they share computed once. Returns each request's masks, every output
    written or found current. A classifier whose every output is current is
    not run.

    A mask the classifier cannot make, or one it finds nowhere, raises
    ``ValueError``; nothing is written then."""
    started = time.perf_counter()
    raw_image_path = Path(raw_image_path)
    loaded: list[tuple[Path, PixelClassifier, dict[str, Path], dict[str, tuple[int, ...]]]] = []
    for classifier_path, outputs in requests:
        classifier_path = Path(classifier_path)
        if not classifier_path.is_file():
            raise FileNotFoundError(f"The classifier {classifier_path} does not exist.")
        classifier = load_classifier(classifier_path)
        try:
            marks = {mask: classifier.mask_labels(mask) for mask in outputs}
        except ValueError as error:
            raise ValueError(f"{classifier_path.name}: {error}") from error
        loaded.append((classifier_path, classifier, {m: Path(p) for m, p in outputs.items()}, marks))
    results = [dict(paths) for _p, _c, paths, _m in loaded]
    if not loaded:
        return results
    shared = shared_spec([classifier.spec for _p, classifier, _o, _m in loaded])
    raw = read_raw_channels(
        raw_image_path, shared.channels, axis_order=axis_order, voxel_size_policy=voxel_size_policy,
        voxel_size_override_xyz=voxel_size_override_xyz, use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    try:
        return _segment(
            raw, loaded, results, shared, started=started, reuse_existing=reuse_existing,
            use_memmap=use_memmap, memmap_directory=memmap_directory, workers=workers, poll=poll,
        )
    finally:
        _release(*raw.volumes)


def _segment(raw, loaded, results, shared, *, started, reuse_existing, use_memmap, memmap_directory, workers, poll):
    from haemolynx.io.ilastik import sidecar_path

    provenances: dict[Path, dict] = {}
    #: Each classifier's stale outputs, by its file: one used for several
    #: jobs (a combined classifier, for the main mask and a role's) walks once.
    grouped: dict[str, tuple[PixelClassifier, list]] = {}
    for classifier_path, classifier, paths, marks in loaded:
        for mask, path in paths.items():
            provenances[path] = classifier_provenance(
                raw.path, classifier_path, mask, channels=classifier.spec.channels,
                voxel_size_zyx=raw.voxel_size_zyx, axis_order=raw.axis_order,
            )
        stale = {
            path: reason
            for path in paths.values()
            for current, reason in [classifier_output_is_current(path, provenances[path])]
            if not (reuse_existing and current)
        }
        if not stale:
            logger.info(
                "Reusing the segmentation of %s by %s: %s.",
                raw.path.name, classifier_path.name, ", ".join(str(p) for p in paths.values()),
            )
            continue
        for path, reason in stale.items():
            if path.exists() and reuse_existing:
                logger.warning("Segmenting %s again: %s.", path, reason)
        key = os.path.normcase(str(classifier_path.resolve()))
        group = grouped.setdefault(key, (classifier, []))
        known = {output for output, _marks in group[1]}
        group[1].extend(
            (str(path), marks[mask]) for mask, path in paths.items() if str(path) not in known
        )
        _warn_about_voxel_size(classifier, raw)
    jobs = [Job(classifier, tuple(outputs)) for classifier, outputs in grouped.values()]
    if not jobs:
        return results
    # Classifiers on the coarse grid and on the image's own share no
    # features: one pass for each (a classifier saved before beside a new one).
    passes: dict[bool, list[Job]] = {}
    for job in jobs:
        passes.setdefault(job.classifier.spec.coarse, []).append(job)
    channels = sorted({c for job in jobs for c in job.classifier.spec.channels}, key=lambda c: -1 if c is None else c)
    volumes = [raw.volume(channel) for channel in channels]
    ranges_by_channel = {channel: intensity_range(raw.volume(channel)) for channel in channels}
    keys = [key for job in jobs for key, _marks in job.outputs]
    logger.info(
        "Segmenting %s with %d classifier(s) in one pass (%s), %s.",
        raw.path.name, len(jobs), ", ".join(Path(k).name for k in keys),
        "read lazily" if not isinstance(volumes[0], np.ndarray) else "in memory",
    )
    last = [0.0]

    def report(done: int, total: int) -> None:
        now = time.perf_counter()
        if now - last[0] > 10.0 or done == total:
            last[0] = now
            logger.info("Classifier: %d of %d boxes (%.0f s).", done, total, now - started)

    streaming = _streams(raw)
    writers: dict[str, _MaskWriter] = {}
    whole: dict[str, np.ndarray] = {}
    if streaming:
        writers = {key: _MaskWriter(Path(key), raw) for key in keys}
    else:
        from haemolynx.preprocessing.memmap_support import new_memmap_array

        for key in keys:
            whole[key] = (
                new_memmap_array(raw.shape_zyx, np.uint8, directory=memmap_directory)
                if use_memmap else np.zeros(raw.shape_zyx, np.uint8)
            )

    def sink(z0, z1, buffers):
        for key, planes in buffers.items():
            if streaming:
                writers[key].put(planes)
            else:
                whole[key][z0:z1] = planes

    budget = LOW_RAM_BUDGET_BYTES
    free = available_memory_bytes()
    if free is not None:
        budget = min(budget, int(0.4 * free)) if use_memmap else max(budget, int(0.4 * free))
    try:
        for group in passes.values():
            used = shared_spec([job.classifier.spec for job in group])
            group_volumes = [raw.volume(channel) for channel in used.channels]
            halo = feature_halo(used, raw.voxel_size_zyx)
            per_slab = slab_planes(
                raw.shape_zyx, halo, channels=len(group_volumes),
                itemsize=int(np.dtype(getattr(group_volumes[0], "dtype", np.uint16)).itemsize),
                outputs=sum(len(job.outputs) for job in group), budget_bytes=budget,
            )
            for volume in group_volumes:
                if hasattr(volume, "cache_bytes"):
                    # The slab's pages with their margin, within the budget:
                    # past it pages are decoded again rather than memory run out.
                    reach = coarse_reach(used, raw.voxel_size_zyx)[0]
                    wanted = (per_slab + 2 * reach + 1) * volume._plane_bytes
                    share = budget // (2 * max(len(group_volumes), 1))
                    volume.cache_bytes = int(min(wanted, max(share, 2 * volume._plane_bytes)))
            classify_slabs(
                group_volumes, [ranges_by_channel[c] for c in used.channels], raw.voxel_size_zyx, group,
                sink=sink, workers=workers, poll=poll, progress=report,
                box_voxels=box_size(budget_bytes=budget, workers=workers), planes_per_slab=per_slab,
            )
    except BaseException:
        for writer in writers.values():
            writer.abort()
        _release(*whole.values())
        raise
    counts: dict[str, int] = {}
    if streaming:
        for key, writer in writers.items():
            sidecar_path(writer.path).unlink(missing_ok=True)
            writer.close()
            counts[key] = writer.vessel
    else:
        for key, mask in whole.items():
            sidecar_path(Path(key)).unlink(missing_ok=True)
            counts[key] = _write_mask(Path(key), mask, raw)
        _release(*whole.values())
    empty = [key for key, count in counts.items() if count == 0]
    if empty:
        for key in keys:
            Path(key).unlink(missing_ok=True)
        raise ValueError(
            f"The classifier found nothing for {', '.join(Path(k).name for k in empty)} in {raw.path.name}: "
            "train it on more of this image, or check the channels it reads."
        )
    voxels = int(np.prod(raw.shape_zyx, dtype=np.int64))
    for key in keys:
        path = Path(key)
        stat = path.stat()
        record = dict(provenances[path])
        record["output"] = {
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "vessel_voxels": counts[key], "voxels": voxels,
        }
        sidecar_path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info("Classifier output %s: %.2f%% of its voxels.", path, 100.0 * counts[key] / voxels)
    logger.info("Classifier segmentation took %.1f s.", time.perf_counter() - started)
    return results


def segment_with_classifier(
    raw_image_path: str | Path,
    classifier_path: str | Path,
    outputs: Mapping[str, str | Path],
    **options,
) -> dict[str, Path]:
    """:func:`segment_with_classifiers` for one classifier: each mask of
    *outputs* (mask name -> output path) written or found current."""
    return segment_with_classifiers(raw_image_path, [(classifier_path, outputs)], **options)[0]


def _release(*arrays) -> None:
    """Close lazy readers, and delete the disk-backed scratch arrays a
    low-RAM segmentation made."""
    from haemolynx.preprocessing.memmap_support import release_memmap_array

    for array in arrays:
        if hasattr(array, "close") and not isinstance(array, np.ndarray):
            array.close()
        elif isinstance(array, np.memmap) and str(getattr(array, "filename", "")).endswith(".memmap"):
            try:
                release_memmap_array(array)
            except OSError:
                pass


def _warn_about_voxel_size(classifier: PixelClassifier, raw: RawChannels) -> None:
    """Warn when the image's voxels are more than twice as large or small,
    along any axis, as every image the classifier was trained on."""
    if not classifier.images:
        return
    for image in classifier.images:
        ratios = np.asarray(raw.voxel_size_zyx) / np.asarray(image.voxel_size_zyx)
        if np.all((ratios <= 2.0) & (ratios >= 0.5)):
            return
    logger.warning(
        "%s's voxels (%s um, z, y, x) are more than twice as large or small as those of every "
        "image the classifier was trained on; its scales are in microns, but check its output.",
        raw.path.name, ", ".join(f"{v:g}" for v in raw.voxel_size_zyx),
    )


def classifier_output_current(
    raw_image_path: str | Path,
    classifier_path: str | Path,
    outputs: Mapping[str, str | Path],
    *,
    axis_order: str = "zyx",
    voxel_size_policy: str = "auto",
    voxel_size_override_xyz=None,
) -> tuple[bool, str]:
    """Whether every output of *outputs* is current (see
    :func:`classifier_output_is_current`), without segmenting or reading
    the raw image's voxels."""
    import tifffile

    from haemolynx.io.axis_order import axis_order_for_file, voxel_size_zyx_from_xyz
    from haemolynx.io.load import _voxel_size_xyz_from_tiff
    from haemolynx.io.voxel_validation import resolve_voxel_size_xyz

    from .model import read_classifier_record

    raw_image_path, classifier_path = Path(raw_image_path), Path(classifier_path)
    channels = FeatureSpec.from_record(read_classifier_record(classifier_path)["spec"]).channels
    order = axis_order_for_file(axis_order, raw_image_path)
    with tifffile.TiffFile(str(raw_image_path)) as tif:
        x, y, z, status = _voxel_size_xyz_from_tiff(tif, order)
    voxel_xyz, _ = resolve_voxel_size_xyz((x, y, z), status, voxel_size_override_xyz, voxel_size_policy)
    for label, path in outputs.items():
        provenance = classifier_provenance(
            raw_image_path, classifier_path, label,
            channels=channels, voxel_size_zyx=voxel_size_zyx_from_xyz(voxel_xyz), axis_order=order,
        )
        current, reason = classifier_output_is_current(path, provenance)
        if not current:
            return False, f"{path}: {reason}"
    return True, ""
