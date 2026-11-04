"""Segmenting a raw image with a saved pixel classifier.

The raw image is read channel by channel in canonical ``(z, y, x)`` order, as
every other reader of it reads it; each box of voxels gets its features
(:mod:`haemolynx.classifier.features`) and its forest's vote, several boxes at
once on worker threads; and each label asked for is written as its own 0/1
TIFF -- in the raw file's own axis order and with its voxel size, so the run
reads it back under the same ``image_axis_order`` as the raw image -- with a
record beside it (``{output}.haemolynx.json``) of how it was made. An output
whose record matches what this run would make is used again rather than
segmented again: the classifier by its bytes, the raw image by path, size,
time and a sample of its bytes, the channels, the label, the voxel size and
the axis order.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .features import boxes, feature_halo, features_of_box, intensity_range
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

#: Voxels per padded box under the low-RAM option: a quarter of the usual,
#: so the boxes in flight hold a quarter of the memory. The answer is the
#: same whatever the box size.
LOW_RAM_BOX_VOXELS = 1_000_000


@dataclass(frozen=True)
class RawChannels:
    """The channels of a raw image a classifier reads, as it reads them."""

    path: Path
    volumes: tuple[np.ndarray, ...]  # canonical (z, y, x), float32
    voxel_size_zyx: tuple[float, float, float]
    axis_order: str  # the order the file was read in
    file_spacing: tuple[float, float, float]  # along the file's own axes

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.volumes[0].shape)  # type: ignore[return-value]


def read_raw_channels(
    path: str | Path,
    channels: Sequence[int | None],
    *,
    axis_order: str = "zyx",
    voxel_size_policy: str = "auto",
    voxel_size_override_xyz=None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> RawChannels:
    """*channels* of the raw TIFF at *path* (``None`` for a file of one
    channel), read as the run reads its raw image, with the voxel size the
    run's voxel-size policy gives it."""
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
        load_channel(path, channel, order, use_memmap=use_memmap, memmap_directory=memmap_directory)
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
) -> np.ndarray:
    """One channel of the raw TIFF at *path* (``None``: a file of one
    channel), in canonical order and in the file's own number type -- the
    features scale it to float32 a box at a time, so no float32 copy of the
    image is ever held. A stack of one plane is not reordered, as the run's
    other readers of a raw image do not reorder it.

    With *use_memmap* (the low-RAM option) the channel is copied a few pages
    at a time into a disk-backed array, never whole in memory. The same
    voxels either way."""
    import tifffile

    from haemolynx.haemodynamics.fwhm_diameter import _read_tiff_channel
    from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER, apply_axis_order
    from haemolynx.io.channels import tiff_channel_axis

    path = Path(path)
    index = 0 if channel is None else int(channel)
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


def classify_volume(
    volumes: Sequence[np.ndarray],
    voxel_size_zyx: Sequence[float],
    classifier: PixelClassifier,
    *,
    ranges: Sequence[tuple[float, float]] | None = None,
    out: np.ndarray | None = None,
    workers: int | None = None,
    poll: Optional[Callable[[], None]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    box_voxels: int | None = None,
) -> np.ndarray:
    """Each voxel's most probable label (uint8, an index into the
    classifier's labels) for the channel *volumes* of one image.

    *ranges* are the channels' intensity ranges (each volume's own by
    default, :func:`~haemolynx.classifier.features.intensity_range`); *out*
    an array to write into (a memory map, for a large image). *poll* is called
    between boxes: whatever it raises stops the work and propagates.
    *progress* hears ``(boxes done, boxes)``.
    """
    from .features import BOX_VOXELS

    if classifier.forest is None:
        raise ValueError("The classifier has not been trained.")
    spec = classifier.spec
    if len(volumes) != len(spec.channels):
        raise ValueError(f"The classifier reads {len(spec.channels)} channel(s); given {len(volumes)}.")
    if ranges is None:
        ranges = [intensity_range(volume) for volume in volumes]
    shape = tuple(int(n) for n in volumes[0].shape)
    if out is None:
        out = np.zeros(shape, dtype=np.uint8)
    box_voxels = int(box_voxels or BOX_VOXELS)
    halo = feature_halo(spec, voxel_size_zyx)
    tiles = list(boxes(shape, halo, box_voxels=box_voxels))
    largest_core = max(int(np.prod([s.stop - s.start for s in core])) for _p, _i, core in tiles)
    count = worker_count(box_voxels, largest_core, spec.n_features, workers=workers)
    forest = classifier.forest

    def one(tile):
        padded, inner, core = tile
        features = features_of_box(volumes, ranges, voxel_size_zyx, spec, padded, inner)
        labels = forest.predict(features.reshape(-1, spec.n_features))
        return core, labels.reshape(features.shape[:3])

    done = 0
    pending: set = set()
    remaining = iter(tiles)
    with ThreadPoolExecutor(max_workers=count) as pool:
        try:
            while True:
                while len(pending) < 2 * count:
                    tile = next(remaining, None)
                    if tile is None:
                        break
                    pending.add(pool.submit(one, tile))
                if not pending:
                    break
                finished, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in finished:
                    core, labels = future.result()
                    out[core] = labels
                    done += 1
                    if progress is not None:
                        progress(done, len(tiles))
                if poll is not None:
                    poll()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return out


def classifier_provenance(
    raw_path: str | Path,
    classifier_path: str | Path,
    label: str,
    *,
    channels: Sequence[int | None],
    voxel_size_zyx: Sequence[float],
    axis_order: str,
) -> dict:
    """What an output of *label* made from *raw_path* by *classifier_path*
    depends on (see the module docstring)."""
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


def count_label(labels: np.ndarray, index: int) -> int:
    """How many voxels of *labels* are *index*, a plane at a time."""
    return int(sum(np.count_nonzero(np.asarray(labels[z]) == index) for z in range(labels.shape[0])))


def _write_mask(path: Path, labels: np.ndarray, index: int, raw: RawChannels) -> None:
    """Where *labels* (canonical order) is *index*, as a 0/1 TIFF in the raw
    file's own axis order, with its voxel size, so it reads back under the
    same axis order as the raw image. Written a plane at a time, so the mask
    is never whole in memory."""
    import tifffile

    from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER, axis_order_transpose

    stored = labels
    if raw.axis_order != CANONICAL_AXIS_ORDER and labels.shape[0] != 1:
        stored = np.transpose(labels, np.argsort(axis_order_transpose(raw.axis_order)))

    def planes():
        for plane in range(stored.shape[0]):
            yield (np.asarray(stored[plane]) == index).astype(np.uint8)

    pages, rows, columns = raw.file_spacing
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".part" + path.suffix)
    tifffile.imwrite(
        partial,
        planes(),
        shape=tuple(int(n) for n in stored.shape),
        dtype=np.uint8,
        imagej=True,
        resolution=(1.0 / columns, 1.0 / rows),
        metadata={"axes": "ZYX", "spacing": pages, "unit": "um"},
        software="HaemoLynx pixel classifier",
    )
    os.replace(partial, path)


def segment_with_classifier(
    raw_image_path: str | Path,
    classifier_path: str | Path,
    outputs: Mapping[str, str | Path],
    *,
    axis_order: str = "zyx",
    voxel_size_policy: str = "auto",
    voxel_size_override_xyz=None,
    reuse_existing: bool = True,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    workers: int | None = None,
    poll: Optional[Callable[[], None]] = None,
) -> dict[str, Path]:
    """Segment the raw image with the classifier saved at *classifier_path*,
    writing each label of *outputs* (label name -> output path) as a 0/1
    mask. Returns the same mapping, every output written or found current.

    The image is classified once for every label asked for. A label the
    classifier was never trained on, or one no voxel was given, raises
    ``ValueError``; nothing is written then.
    """
    started = time.perf_counter()
    raw_image_path = Path(raw_image_path)
    classifier_path = Path(classifier_path)
    if not classifier_path.is_file():
        raise FileNotFoundError(f"The classifier {classifier_path} does not exist.")
    classifier = load_classifier(classifier_path)
    trained = classifier.trained_labels()
    for label in outputs:
        if label not in classifier.labels:
            raise ValueError(f"{classifier_path.name} has no label {label!r}; its labels are {list(classifier.labels)}.")
        if label not in trained:
            raise ValueError(f"{classifier_path.name} was never shown any {label} voxels, so it cannot find them.")
    raw = read_raw_channels(
        raw_image_path,
        classifier.spec.channels,
        axis_order=axis_order,
        voxel_size_policy=voxel_size_policy,
        voxel_size_override_xyz=voxel_size_override_xyz,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    provenances = {
        label: classifier_provenance(
            raw_image_path,
            classifier_path,
            label,
            channels=classifier.spec.channels,
            voxel_size_zyx=raw.voxel_size_zyx,
            axis_order=raw.axis_order,
        )
        for label in outputs
    }
    paths = {label: Path(path) for label, path in outputs.items()}
    if reuse_existing:
        stale = {}
        for label, path in paths.items():
            current, reason = classifier_output_is_current(path, provenances[label])
            if not current:
                stale[label] = reason
        if not stale:
            logger.info(
                "Reusing the classifier's segmentation of %s: %s (made from this image by this "
                "classifier, with the same settings).",
                raw_image_path.name,
                ", ".join(str(path) for path in paths.values()),
            )
            return paths
        for label, reason in stale.items():
            if paths[label].exists():
                logger.warning("Segmenting %s again: %s.", paths[label], reason)
    _warn_about_voxel_size(classifier, raw)
    out = None
    if use_memmap:
        from haemolynx.preprocessing.memmap_support import new_memmap_array

        out = new_memmap_array(raw.shape_zyx, np.uint8, directory=memmap_directory)
    logger.info(
        "Segmenting %s with the classifier %s (%s; %d features).",
        raw_image_path.name, classifier_path.name, ", ".join(outputs), classifier.spec.n_features,
    )
    last = [0.0]

    def report(done: int, total: int) -> None:
        now = time.perf_counter()
        if now - last[0] > 10.0 or done == total:
            last[0] = now
            logger.info("Classifier: %d of %d boxes (%.0f s).", done, total, now - started)

    labels = classify_volume(
        raw.volumes, raw.voxel_size_zyx, classifier, out=out, workers=workers, poll=poll, progress=report,
        box_voxels=LOW_RAM_BOX_VOXELS if use_memmap else None,
    )
    from haemolynx.io.ilastik import sidecar_path

    counts = {label: count_label(labels, classifier.label_index(label)) for label in paths}
    for label, vessel in counts.items():
        if vessel == 0:
            raise ValueError(
                f"The classifier found no {label} voxels in {raw_image_path.name}: train it on more of "
                "this image, or check the channels it reads."
            )
    voxels = int(labels.size)
    for label, path in paths.items():
        sidecar_path(path).unlink(missing_ok=True)
        _write_mask(path, labels, classifier.label_index(label), raw)
        stat = path.stat()
        record = dict(provenances[label])
        record["output"] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "vessel_voxels": counts[label],
            "voxels": voxels,
        }
        sidecar_path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info(
            "Classifier output %s: %.1f%% of its voxels are %s.", path, 100.0 * counts[label] / voxels, label
        )
    _release(labels, *raw.volumes)
    logger.info("Classifier segmentation took %.1f s.", time.perf_counter() - started)
    return paths


def _release(*arrays) -> None:
    """Delete the disk-backed scratch arrays a low-RAM segmentation made."""
    from haemolynx.preprocessing.memmap_support import release_memmap_array

    for array in arrays:
        if isinstance(array, np.memmap) and str(getattr(array, "filename", "")).endswith(".memmap"):
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

    from .features import FeatureSpec
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
