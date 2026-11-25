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
    intensity_range_and_profile,
)
from .model import PixelClassifier, load_classifier
from .slabs import SlabStore, full_size_planes

logger = logging.getLogger(__name__)

#: What a segmentation's record says made it.
SOURCE = "HaemoLynx pixel classifier"

#: The record's layout; a record of any other is not trusted.
PROVENANCE_VERSION = 1

#: The fields that must match this run's for an output to be used again.
_PROVENANCE_KEYS = (
    "version", "source", "classifier_sha256", "input", "label", "voxel_size_zyx", "axis_order", "grid",
)

#: Why an output is not current, per field of its record that differs.
_PROVENANCE_REASONS = {
    "version": "it was recorded by another version of HaemoLynx",
    "source": "it was not made by a HaemoLynx classifier",
    "classifier_sha256": "the classifier is not the one it was made with",
    "input": "the raw image, or the channels read of it, has changed",
    "label": "it is another label's mask",
    "voxel_size_zyx": "the raw image's voxel size has changed",
    "axis_order": "the raw image is read in another axis order",
    "grid": "it was classified on another grid",
}

#: About how much memory one box being worked on takes, per padded voxel and
#: per feature of a core voxel (bytes): the scaled channels and a scale's
#: smoothed image and derivatives, then the features and the forest's walk.
_BYTES_PER_PADDED_VOXEL = 64
_BYTES_PER_CORE_FEATURE = 12

#: Never more workers than this, however much memory there is: past it the
#: threads wait on one another.
MAX_WORKERS = 8

#: Never more worker processes than this: each its own interpreter, so they
#: run side by side, as threads do not.
MAX_PROCESSES = 32

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
#: small part of them -- but cut down (:func:`tile_side`) until there are
#: enough of them to keep every worker busy.
COARSE_TILE = 1024
MIN_COARSE_TILE = 256
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


@dataclass
class _Pass:
    """What every tile of a pass reads: sent once to each worker process."""

    volumes: list
    ranges: list
    voxel_size_zyx: tuple
    shared: FeatureSpec
    classifiers: list
    outputs: list  # per job: ((key, marks), ...)
    columns: list
    needed: np.ndarray
    halo: tuple
    shape: tuple
    tiled: bool
    #: Worker processes sharing the decoded-page budget.
    share: int = 1


def _run_tile(state: _Pass, region: tuple, cores: list) -> dict[str, np.ndarray]:
    """Every output of *region* (a tile of a slab): its coarse context, then
    its boxes one after another."""
    context = (
        coarse_context(state.volumes, state.ranges, state.voxel_size_zyx, state.shared, region, state.needed)
        if state.tiled else None
    )
    extent = tuple(r.stop - r.start for r in region)
    out = {
        key: np.zeros(extent, dtype=np.uint8)
        for outputs in state.outputs for key, _marks in outputs
    }
    nf = state.shared.n_features
    for core in cores:
        padded = tuple(
            slice(max(0, c.start - h), min(n, c.stop + h)) for c, h, n in zip(core, state.halo, state.shape)
        )
        inner = tuple(slice(c.start - p.start, c.stop - p.start) for c, p in zip(core, padded))
        features = features_of_box(
            state.volumes, state.ranges, state.voxel_size_zyx, state.shared, padded, inner,
            needed=state.needed, coarse=context,
        )
        flat = features.reshape(-1, nf)
        local = tuple(slice(c.start - r.start, c.stop - r.start) for c, r in zip(core, region))
        for classifier, outputs, cols in zip(state.classifiers, state.outputs, state.columns):
            labels = classifier.forest.predict(np.ascontiguousarray(flat[:, cols])).reshape(features.shape[:3])
            for key, marks in outputs:
                if marks is None:
                    out[key][local] = labels
                elif len(marks) == 1:
                    out[key][local] = labels == marks[0]
                else:
                    out[key][local] = np.isin(labels, marks)
    return out


#: The pass a worker process was started for (:func:`_start_worker`).
_WORKER_PASS: list = []


def _page_reader(volume):
    """The reader keeping *volume*'s decoded pages (a lazily read channel,
    or the one a block-mean view reads through); None for an array."""
    reader = volume
    while reader is not None and not hasattr(reader, "cache_bytes"):
        reader = getattr(reader, "volume", None)
    return reader


def _start_worker(state: _Pass) -> None:
    for volume in state.volumes:
        reader = _page_reader(volume)
        if reader is not None:
            reader.cache_bytes = max(reader.cache_bytes // max(state.share, 1), 2 * getattr(reader, "_plane_bytes", 0))
    _WORKER_PASS[:] = [state]


def _worker_tile(region: tuple, cores: list) -> dict[str, np.ndarray]:
    return _run_tile(_WORKER_PASS[0], region, cores)


#: Images with fewer voxels than this are classified on threads: starting
#: worker processes costs more than it saves.
PROCESS_MIN_VOXELS = 64_000_000


def _can_use_processes(volumes) -> bool:
    """Whether every volume can be sent to a worker process as where to read
    it, not as its voxels."""
    from .lazy import LazyTiffChannel

    def light(volume):
        if isinstance(volume, LazyTiffChannel):
            return True
        inner = getattr(volume, "volume", None)
        return inner is not None and light(inner)

    return all(light(volume) for volume in volumes)


def process_count(n_features: int, box_voxels: int, *, processes: int | None = None) -> int:
    """How many worker processes: *processes* when given, else one a core
    (at most :data:`MAX_PROCESSES`), fewer where free memory would not hold
    their boxes, tiles and decoded pages."""
    if processes is not None:
        return max(1, int(processes))
    # One core left for this process, which gathers the tiles and writes.
    cores = max(1, (os.cpu_count() or 2) - 1)
    per_process = box_voxels * _BYTES_PER_PADDED_VOXEL + box_voxels * n_features * 4
    free = available_memory_bytes()
    by_memory = MAX_PROCESSES if free is None else max(1, int(0.7 * free // max(per_process, 1)))
    return max(1, min(cores, MAX_PROCESSES, by_memory))


def tile_side(shape, slabs: int, workers: int) -> int:
    """The side (voxels) of the tiles whose coarse context their boxes share:
    :data:`COARSE_TILE` at most, smaller (to :data:`MIN_COARSE_TILE`, in steps
    of 64) until the image holds a few tiles a worker, so none sits idle."""
    area = int(shape[1]) * int(shape[2])
    side = int(np.sqrt(area * max(slabs, 1) / max(3 * workers, 1)))
    side = (side // 64) * 64
    return int(max(MIN_COARSE_TILE, min(COARSE_TILE, side)))


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
    processes: bool | None = None,
    start_plane: int = 0,
) -> None:
    """Classify the image a slab of planes at a time for every job at once,
    handing *sink* each slab's outputs, ``(z0, z1, {key: uint8 (planes, y, x)})``,
    in order. *volumes* and *ranges* are the shared spec's channels
    (:func:`shared_spec` over the jobs'), in its order. *poll* is called
    between tiles: whatever it raises stops the work and propagates.
    *progress* hears ``(tiles done, tiles)``. *start_plane*: the planes
    before it are not classified (a segmentation taken up where it stopped).

    The work is cut into tiles of a slab -- a box each, or, with coarse
    scales, a large tile whose coarse context its boxes share -- run on worker
    processes (*processes*; by default for an image of at least
    :data:`PROCESS_MIN_VOXELS` read lazily, each process with its own
    interpreter, so they run side by side) or threads. Several slabs may be
    in flight; *sink* still hears them in order. The labels are the same
    every way."""
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
    slabs = [
        (z0, min(z0 + planes_per_slab, shape[0])) for z0 in range(int(start_plane), shape[0], planes_per_slab)
    ]
    if not slabs:
        return
    tiled = has_coarse_scales(shared, voxel_size_zyx)
    use_processes = processes
    if use_processes is None:
        use_processes = int(np.prod(shape, dtype=np.int64)) >= PROCESS_MIN_VOXELS and _can_use_processes(volumes)
    if use_processes and not _can_use_processes(volumes):
        use_processes = False
    planned = (
        process_count(int(needed.sum()), box_voxels, processes=workers) if use_processes
        else worker_count(box_voxels, box_voxels, int(needed.sum()), workers=workers)
    )
    side = tile_side(shape, len(slabs), planned)
    tasks = []  # (slab number, region, cores)
    for number, (z0, z1) in enumerate(slabs):
        by, bx = _slab_core(shape, halo, (z0, z1), box_voxels)
        tile = (side, side) if tiled else (by, bx)
        for ty in range(0, shape[1], tile[0]):
            for tx in range(0, shape[2], tile[1]):
                region = (slice(z0, z1), slice(ty, min(ty + tile[0], shape[1])), slice(tx, min(tx + tile[1], shape[2])))
                cores = [
                    (slice(z0, z1), slice(y0, min(y0 + by, region[1].stop)), slice(x0, min(x0 + bx, region[2].stop)))
                    for y0 in range(ty, region[1].stop, by)
                    for x0 in range(tx, region[2].stop, bx)
                ]
                tasks.append((number, region, cores))
    state = _Pass(
        volumes=list(volumes), ranges=[tuple(r) for r in ranges], voxel_size_zyx=tuple(voxel_size_zyx),
        shared=shared, classifiers=[_without_training(job.classifier) for job in jobs],
        outputs=[tuple(job.outputs) for job in jobs], columns=columns, needed=needed, halo=halo,
        shape=shape, tiled=tiled,
    )
    largest = max(int(np.prod([s.stop - s.start for s in core])) for _n, _r, cores in tasks for core in cores)
    if use_processes:
        from concurrent.futures import ProcessPoolExecutor

        count = planned
        state.share = count
        pool = ProcessPoolExecutor(max_workers=count, initializer=_start_worker, initargs=(state,))
        run = _worker_tile
    else:
        count = worker_count(box_voxels, largest, int(needed.sum()), workers=workers)
        pool = ThreadPoolExecutor(max_workers=count)

        def run(region, cores):
            return _run_tile(state, region, cores)

    keys = [key for job in jobs for key, _marks in job.outputs]
    remaining = [0] * len(slabs)
    for number, _region, _cores in tasks:
        remaining[number] += 1
    buffers: dict[int, dict[str, np.ndarray]] = {}
    next_sink = 0
    done = 0
    pending: dict = {}
    queued = iter(tasks)
    try:
        while True:
            while len(pending) < 2 * count:
                task = next(queued, None)
                if task is None:
                    break
                number, region, cores = task
                pending[pool.submit(run, region, cores)] = (number, region)
            if not pending:
                break
            finished, _ = wait(list(pending), timeout=0.5, return_when=FIRST_COMPLETED)
            for future in finished:
                number, region = pending.pop(future)
                result = future.result()
                z0, z1 = slabs[number]
                slab = buffers.setdefault(
                    number, {key: np.zeros((z1 - z0,) + shape[1:], dtype=np.uint8) for key in keys}
                )
                local = (slice(0, z1 - z0),) + region[1:]
                for key, values in result.items():
                    slab[key][local] = values
                remaining[number] -= 1
                done += 1
                if progress is not None:
                    progress(done, len(tasks))
            while next_sink < len(slabs) and remaining[next_sink] == 0:
                z0, z1 = slabs[next_sink]
                sink(z0, z1, buffers.pop(next_sink))
                next_sink += 1
            if poll is not None:
                poll()
    except BaseException:
        # Stopped (or failed): no waiting for tiles still running.
        for future in pending:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


def _without_training(classifier: PixelClassifier) -> PixelClassifier:
    """The classifier as a pass needs it: its forest, not its training voxels."""
    from dataclasses import replace as replaced

    return replaced(classifier, training=None)


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
    processes: bool | None = None,
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
        processes=processes,
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
    grid: Sequence[int] | None = None,
) -> dict:
    """What an output of mask *label* made from *raw_path* by
    *classifier_path* depends on (see the module docstring), and the coarser
    *grid* it was classified on, when it was."""
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
    if grid is not None and any(int(f) > 1 for f in grid):
        record["grid"] = [int(f) for f in grid]
    return record


def classifier_output_is_current(output_path: str | Path, provenance: dict) -> tuple[bool, str]:
    """Whether the output at *output_path* was made as *provenance* says
    this run would make it, and why not: the rule ilastik's outputs are
    reused by (``io.ilastik.recorded_output_is_current``), on this record."""
    from haemolynx.io.ilastik import recorded_output_is_current

    return recorded_output_is_current(output_path, provenance, {key: _PROVENANCE_REASONS[key] for key in _PROVENANCE_KEYS})


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
        #: Whether the writer has taken the end (None) off the queue.
        self._ended = False
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = raw.shape_zyx

        def planes():
            while True:
                plane = self._planes.get()
                if plane is None:
                    self._ended = True
                    return
                yield plane

        def write():
            try:
                tifffile.imwrite(self.partial, planes(), shape=shape, dtype=np.uint8, **_mask_metadata(raw))
            except BaseException as error:  # noqa: BLE001 - raised on close
                self._error.append(error)
                # Drained up to the end, so put() never blocks -- unless the
                # planes already took it (abort() before the last plane).
                while not self._ended and self._planes.get() is not None:
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


def read_written_mask(
    path: str | Path,
    axis_order: str,
    shape_zyx: Sequence[int],
    *,
    lazy: bool = False,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
):
    """A mask a segmentation wrote, back in canonical ``(z, y, x)`` order: it
    was stored in the raw file's own *axis_order* (the order the raw image
    of *shape_zyx*, canonical, was read in), so it is read in that order, as
    the run reads it -- z first, as a pass streams it, when the order is
    canonical or the image one plane deep (:func:`_streams`). *lazy* reads it
    where it is looked at, as :func:`load_channel` can."""
    from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER

    streamed = axis_order == CANONICAL_AXIS_ORDER or int(shape_zyx[0]) == 1
    volume = load_channel(
        path, None, CANONICAL_AXIS_ORDER if streamed else axis_order,
        use_memmap=use_memmap, memmap_directory=memmap_directory, lazy=lazy,
    )
    return volume[np.newaxis] if getattr(volume, "ndim", 3) == 2 else volume


# ----------------------------------------------------------- segmentation

#: How often, at most, a segmentation tells its *progress* how far it is (s).
PROGRESS_SECONDS = 1.0


@dataclass(frozen=True)
class SegmentProgress:
    """How far a segmentation has got (``segment_with_classifiers(progress=)``).

    *done* of *total* tiles of the image classified in this pass (of
    *passes*: ``(this one, how many)``), *seconds* since it began, and
    *remaining*: about how many seconds are left, None until it can be told.
    *start_plane*: the planes before it were classified by a segmentation
    that stopped there, and are kept. *writing*: classifying is over, and the
    masks are being put together."""

    done: int
    total: int
    seconds: float
    remaining: float | None
    passes: tuple[int, int] = (1, 1)
    start_plane: int = 0
    writing: bool = False


def remaining_seconds(done: int, total: int, seconds: float, counted: int) -> float | None:
    """About how many seconds are left with *done* of *total* tiles finished,
    from the *counted* tiles finished in the last *seconds* (timed from the
    first tile's end, so the workers' start-up is not counted as work)."""
    if counted <= 0 or seconds <= 0:
        return None
    return seconds / counted * max(0, total - done)


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
    downsample=1,
    progress: Optional[Callable[[SegmentProgress], None]] = None,
) -> list[dict[str, Path]]:
    """Segment the raw image with each classifier of *requests* --
    ``(classifier path, {mask: output path})`` -- in one pass, the features
    they share computed once. Returns each request's masks, every output
    written or found current. A classifier whose every output is current is
    not run.

    A mask the classifier cannot make raises ``ValueError`` before anything
    is read. One it finds nowhere raises once the pass is done: that mask is
    removed, and every mask it did find is kept, recorded and used again.

    Each mask is kept a slab at a time beside its output while it is made
    (:mod:`~haemolynx.classifier.slabs`): stopped part-way (*poll* raising,
    or the process ending), a later segmentation making it the same way,
    with *reuse_existing*, classifies only the planes after the last slab
    kept, and the masks are the same bit for bit. *progress* hears a
    :class:`SegmentProgress` at most every :data:`PROGRESS_SECONDS`. An image
    whose intensities are unlike every image a classifier was trained on is
    warned about (:mod:`~haemolynx.classifier.familiar`).

    *downsample* (``pixel_classifier_downsample``: ``"2x"``, 4, ...)
    classifies on a coarser grid -- the raw image's block means, the coarser
    axes reduced less (``io.downsample.axis_downsample_factors``) -- and
    writes each mask back at the image's own size, a block of voxels per
    voxel classified: at 2x, a fraction of the work, at the coarser grid's
    resolution."""
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
    if reuse_existing and _all_outputs_current(
        raw_image_path, loaded, axis_order=axis_order, voxel_size_policy=voxel_size_policy,
        voxel_size_override_xyz=voxel_size_override_xyz, downsample=downsample,
    ):
        # Decided from the records and the raw file's header: its voxels
        # are not read just to find nothing to do.
        logger.info("Reusing the segmentation of %s: every mask asked for is current.", raw_image_path.name)
        return results
    raw = read_raw_channels(
        raw_image_path, shared.channels, axis_order=axis_order, voxel_size_policy=voxel_size_policy,
        voxel_size_override_xyz=voxel_size_override_xyz, use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    factors = grid_factors(downsample, raw.voxel_size_zyx)
    grid = raw
    if factors != (1, 1, 1):
        from .lazy import BlockMeanVolume

        grid = RawChannels(
            path=raw.path,
            volumes=tuple(BlockMeanVolume(volume, factors) for volume in raw.volumes),
            voxel_size_zyx=tuple(v * f for v, f in zip(raw.voxel_size_zyx, factors)),
            axis_order=raw.axis_order, file_spacing=raw.file_spacing, channels=raw.channels,
        )
    try:
        return _segment(
            raw, loaded, results, shared, started=started, reuse_existing=reuse_existing,
            use_memmap=use_memmap, memmap_directory=memmap_directory, workers=workers, poll=poll,
            grid=grid, factors=factors, progress=progress,
        )
    finally:
        _release(*raw.volumes)


def _all_outputs_current(raw_image_path: Path, loaded, **reading) -> bool:
    """Whether every output of every loaded request is current
    (:func:`classifier_output_current`: the records and the raw file's
    header, never its voxels). False when that cannot be told here; the
    pass then reads the image and decides, as it always did."""
    if raw_image_path.suffix.lower() not in (".tif", ".tiff"):
        return False
    try:
        return all(
            classifier_output_current(raw_image_path, classifier_path, paths, **reading)[0]
            for classifier_path, _classifier, paths, _marks in loaded
        )
    except (OSError, ValueError, KeyError):
        return False


def classifier_downsample(settings: Mapping[str, object]) -> str:
    """The grid every classifier pass of a run works on: its
    ``pixel_classifier_downsample`` while the main image's classifier switch,
    which that setting hangs under, is on; ``"none"`` otherwise."""
    if not settings.get("use_pixel_classifier_segmentation"):
        return "none"
    return str(settings.get("pixel_classifier_downsample") or "none")


def grid_factors(downsample, voxel_size_zyx) -> tuple[int, int, int]:
    """The coarser grid *downsample* asks the classifier to work on, per
    axis, as the run's own downsampling chooses them; (1, 1, 1) for none."""
    from haemolynx.io.downsample import axis_downsample_factors, downsample_factor

    factor = downsample_factor(downsample)
    if factor == 1:
        return (1, 1, 1)
    return tuple(int(f) for f in axis_downsample_factors(factor, voxel_size_zyx))  # type: ignore[return-value]


def _resume_point(stores: Sequence[SlabStore]) -> int:
    """The grid plane every one of *stores* has its slabs up to, those past
    it dropped: where a pass writing them all takes up."""
    while True:
        start = min(store.finished_planes() for store in stores)
        for store in stores:
            store.keep_before(start)
        if all(store.finished_planes() == start for store in stores):
            return start


def _segment(
    raw, loaded, results, shared, *, started, reuse_existing, use_memmap, memmap_directory, workers, poll,
    grid=None, factors=(1, 1, 1), progress=None,
):
    """*raw* is the image as written to; *grid* as classified (the same, or
    its block means on a grid *factors* coarser)."""
    from haemolynx.io.ilastik import sidecar_path

    grid = grid or raw

    provenances: dict[Path, dict] = {}
    #: Each classifier's stale outputs, by its file: one used for several
    #: jobs (a combined classifier, for the main mask and a role's) walks once.
    grouped: dict[str, tuple[PixelClassifier, list]] = {}
    for classifier_path, classifier, paths, marks in loaded:
        for mask, path in paths.items():
            provenances[path] = classifier_provenance(
                raw.path, classifier_path, mask, channels=classifier.spec.channels,
                voxel_size_zyx=raw.voxel_size_zyx, axis_order=raw.axis_order, grid=factors,
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
    volumes = [grid.volume(channel) for channel in channels]
    read = {channel: intensity_range_and_profile(grid.volume(channel)) for channel in channels}
    ranges_by_channel = {channel: value_range for channel, (value_range, _profile) in read.items()}
    # Compared at the image's own size, as the training images were read.
    profiles = (
        {channel: profile for channel, (_range, profile) in read.items()} if factors == (1, 1, 1)
        else {channel: intensity_range_and_profile(raw.volume(channel))[1] for channel in channels}
    )
    _warn_if_unfamiliar(loaded, grouped, profiles, raw)
    keys = [key for job in jobs for key, _marks in job.outputs]
    logger.info(
        "Segmenting %s with %d classifier(s) in one pass (%s), %s.",
        raw.path.name, len(jobs), ", ".join(Path(k).name for k in keys),
        "read lazily" if not isinstance(volumes[0], np.ndarray) else "in memory",
    )

    def reporter(number: int, start_plane: int):
        """The *number*-th pass's progress, to the log and to *progress*."""
        begun = time.perf_counter()
        first: list = []  # (time, tiles done) as the first tile finished
        logged = [0.0]
        told = [0.0]

        def report(done: int, total: int) -> None:
            now = time.perf_counter()
            if not first:
                first.append((now, done))
            if now - logged[0] > 10.0 or done == total:
                logged[0] = now
                logger.info("Classifier: %d of %d boxes (%.0f s).", done, total, now - started)
            if progress is not None and (now - told[0] >= PROGRESS_SECONDS or done == total):
                told[0] = now
                progress(SegmentProgress(
                    done=done, total=total, seconds=now - begun,
                    remaining=remaining_seconds(done, total, now - first[0][0], done - first[0][1]),
                    passes=(number, len(passes)), start_plane=start_plane,
                ))

        return report

    streaming = _streams(raw)
    stores: dict[str, SlabStore] = {}
    whole: dict[str, np.ndarray] = {}
    if streaming:
        # Each mask kept a slab at a time beside its output: one stopped
        # part-way is taken up where it stopped (classifier.slabs).
        stores = {
            key: SlabStore(
                Path(key), provenances[Path(key)], keep=reuse_existing, factors=factors,
                shape_zyx=raw.shape_zyx, compression=MASK_COMPRESSION,
            )
            for key in keys
        }
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
                stores[key].put(z0, z1, planes)
            elif factors == (1, 1, 1):
                whole[key][z0:z1] = planes
            else:
                for z, plane in full_size_planes(z0, planes, factors, raw.shape_zyx):
                    whole[key][z] = plane

    budget = LOW_RAM_BUDGET_BYTES
    free = available_memory_bytes()
    if free is not None:
        budget = min(budget, int(0.4 * free)) if use_memmap else max(budget, int(0.4 * free))
    try:
        for number, group in enumerate(passes.values(), start=1):
            used = shared_spec([job.classifier.spec for job in group])
            group_volumes = [grid.volume(channel) for channel in used.channels]
            halo = feature_halo(used, grid.voxel_size_zyx)
            per_slab = slab_planes(
                grid.shape_zyx, halo, channels=len(group_volumes),
                itemsize=int(np.dtype(getattr(group_volumes[0], "dtype", np.uint16)).itemsize),
                outputs=sum(len(job.outputs) for job in group), budget_bytes=budget,
            )
            for volume in group_volumes:
                reader = _page_reader(volume)
                if reader is not None:
                    # The slab's pages with their margin, within the budget:
                    # past it pages are decoded again rather than memory run
                    # out. A grid plane spans factors[0] of the image's.
                    reach = coarse_reach(used, grid.voxel_size_zyx)[0]
                    wanted = (per_slab + 2 * reach + 1) * factors[0] * reader._plane_bytes
                    share = budget // (2 * max(len(group_volumes), 1))
                    reader.cache_bytes = int(min(wanted, max(share, 2 * reader._plane_bytes)))
            start = 0
            if streaming:
                start = _resume_point([stores[key] for job in group for key, _marks in job.outputs])
                if start:
                    logger.info(
                        "Taking up the segmentation of %s at plane %d of %d, where one stopped.",
                        raw.path.name, start, grid.shape_zyx[0],
                    )
            classify_slabs(
                group_volumes, [ranges_by_channel[c] for c in used.channels], grid.voxel_size_zyx, group,
                sink=sink, workers=workers, poll=poll, progress=reporter(number, start),
                box_voxels=box_size(budget_bytes=budget, workers=workers), planes_per_slab=per_slab,
                start_plane=start,
            )
        for store in stores.values():
            store.close()
    except BaseException:
        # What was classified stays beside each output, to be taken up.
        for store in stores.values():
            try:
                store.close()
            except Exception:  # noqa: BLE001 - the error that stopped the pass is the one raised
                pass
        _release(*whole.values())
        raise
    counts: dict[str, int] = {}
    if streaming:
        if progress is not None:
            elapsed = time.perf_counter() - started
            progress(SegmentProgress(done=1, total=1, seconds=elapsed, remaining=None, writing=True))
        workers_for_writing = max(1, min(len(keys), 4))
        with ThreadPoolExecutor(max_workers=workers_for_writing) as writing:
            made = list(writing.map(lambda key: _assemble(stores[key], raw, grid.shape_zyx[0]), keys))
        counts = dict(zip(keys, made))
    else:
        for key, mask in whole.items():
            sidecar_path(Path(key)).unlink(missing_ok=True)
            counts[key] = _write_mask(Path(key), mask, raw)
        _release(*whole.values())
    empty = [key for key, count in counts.items() if count == 0]
    voxels = int(np.prod(raw.shape_zyx, dtype=np.int64))
    for key in keys:
        if key in empty:
            # Only the empty mask goes: the others are whole, and used again.
            Path(key).unlink(missing_ok=True)
            continue
        path = Path(key)
        stat = path.stat()
        record = dict(provenances[path])
        record["output"] = {
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "vessel_voxels": counts[key], "voxels": voxels,
        }
        sidecar_path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info("Classifier output %s: %.2f%% of its voxels.", path, 100.0 * counts[key] / voxels)
    for store in stores.values():
        store.remove()
    if empty:
        raise ValueError(
            f"The classifier found nothing for {', '.join(Path(k).name for k in empty)} in {raw.path.name}: "
            "train it on more of this image, or check the channels it reads. The masks it did find are "
            "kept, and used again."
        )
    logger.info("Classifier segmentation took %.1f s.", time.perf_counter() - started)
    return results


def _assemble(store: SlabStore, raw: RawChannels, grid_depth: int) -> int:
    """The output of *store* put together from its slabs; its vessel voxels.
    Every grid plane must be in a slab."""
    from haemolynx.io.ilastik import sidecar_path

    if store.finished_planes() != grid_depth:
        raise RuntimeError(
            f"The slabs of {store.output.name} in {store.folder} stop at plane {store.finished_planes()} "
            f"of {grid_depth}: segment again."
        )
    writer = _MaskWriter(store.output, raw)
    try:
        for plane in store.planes():
            writer.put(plane[np.newaxis])
    except BaseException:
        writer.abort()
        raise
    sidecar_path(store.output).unlink(missing_ok=True)
    writer.close()
    return writer.vessel


def _warn_if_unfamiliar(loaded, grouped, profiles, raw: RawChannels) -> None:
    """Warn, once per classifier about to run, when the image is unlike every
    image it was trained on (:mod:`~haemolynx.classifier.familiar`)."""
    from .familiar import unfamiliar_channels, unfamiliar_note

    warned: set[str] = set()
    for classifier_path, classifier, _paths, _marks in loaded:
        key = os.path.normcase(str(classifier_path.resolve()))
        if key not in grouped or key in warned:
            continue
        warned.add(key)
        found = unfamiliar_channels(classifier, profiles)
        if found:
            logger.warning(unfamiliar_note(raw.path.name, classifier_path.name, found))


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
    downsample=1,
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
            grid=grid_factors(downsample, voxel_size_zyx_from_xyz(voxel_xyz)),
        )
        current, reason = classifier_output_is_current(path, provenance)
        if not current:
            return False, f"{path}: {reason}"
    return True, ""
