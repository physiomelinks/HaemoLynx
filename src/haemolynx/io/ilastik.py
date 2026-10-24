"""Utilities for running ilastik segmentation in headless mode.

ilastik's "Simple Segmentation" export gives each voxel the number of its most
probable label, 1 to N in the order the project's labels were made. Which of
those is the vessel is the project's business, not something the export's
values can tell: a two-label output is ``{1, 2}`` whichever label the vessels
are, and where vessels fill most of a volume the minority is the background.
So the vessel label is named (:func:`resolve_vessel_label`, against the label
names the ``.ilp`` holds), and each output is rewritten as a 0/1 mask
(:func:`binarise_ilastik_output`) that every reader of a segmented image reads
the same way.

Whether an output already made can stand for this run is decided by a sidecar
written beside it (``<output>.haemolynx.json``, :func:`ilastik_provenance`):
the project's SHA-256, the input image's path, size and modification time,
and the vessel label. File times alone could not decide it: copying a file
keeps its modification time on Windows, so a colleague's project copied in,
older than the output, was never run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: How often (seconds) the wait loop wakes to check for cancellation, the
#: timeout and new output while ilastik runs.
POLL_INTERVAL_SECONDS = 0.5

#: How many of ilastik's last output lines a failure message quotes.
_TAIL_LINES = 40

#: What ilastik is asked to export: each voxel the 1-based number of its most
#: probable label.
EXPORT_SOURCE = "Simple Segmentation"

#: ilastik's export format for each output suffix, and the extension ilastik
#: gives that format's files. Without ``--output_format`` ilastik writes the
#: format saved in the project (HDF5 unless changed there), and it replaces
#: whatever extension it is given with the format's own: ``.tif`` comes back
#: ``.tiff``.
OUTPUT_FORMATS = {
    ".tif": ("multipage tiff", ".tiff"),
    ".tiff": ("multipage tiff", ".tiff"),
    ".h5": ("hdf5", ".h5"),
}

#: Where a pixel-classification project keeps its label names, label 1 first.
LABEL_NAMES_DATASET = "PixelClassification/LabelNames"

#: Appended to an output's file name for the sidecar recording how it was made.
SIDECAR_SUFFIX = ".haemolynx.json"

#: The sidecar's layout. A sidecar of any other version is not trusted.
PROVENANCE_VERSION = 1

#: The fields of a sidecar that must match this run's for its output to be
#: reused (the rest are there for a person reading it).
_PROVENANCE_KEYS = ("version", "export_source", "project_sha256", "input", "vessel_label")

_SHA256_CHUNK_BYTES = 8 * 1024 * 1024

#: SHA-256 of files already hashed in this process, keyed on path, size and
#: modification time, so one project is hashed once per run, not per mask.
_SHA256_MEMO: dict[tuple[str, int, int], str] = {}


def ilastik_label_names(project_path: str | Path) -> list[str] | None:
    """The label names a pixel-classification project was trained with,
    label 1 first, read from the ``.ilp`` (an HDF5 file). None when there are
    none to read: not an HDF5 file, or not a pixel-classification project."""
    try:
        import h5py

        with h5py.File(project_path, "r") as project:
            names = project[LABEL_NAMES_DATASET][()]
    except (OSError, KeyError, ValueError, TypeError):
        return None
    return [
        name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
        for name in np.atleast_1d(names).tolist()
    ]


def describe_labels(names: Sequence[str]) -> str:
    """``1 ('Vessel'), 2 ('Background')``: each label as a setting may name it."""
    return ", ".join(f"{number} ({name!r})" for number, name in enumerate(names, start=1))


def resolve_vessel_label(
    label: object,
    names: Sequence[str] | None,
    *,
    setting: str = "vessel_label",
    project: str | Path | None = None,
) -> int:
    """The number (1-based) the export gives the vessel label.

    *label* is one of the project's label *names* (exactly, else ignoring
    case when that picks one), or a label's number. With *names* None (the
    project's could not be read) only a number is accepted. Anything else
    raises ``ValueError`` naming *setting* and listing the labels there are.
    """
    where = f" in {Path(project).name}" if project is not None else ""
    choices = (
        f"one of the project's labels: {describe_labels(names)}"
        if names
        else "a label number (1 for the first label made in ilastik)"
    )
    text = "" if label is None else str(label).strip()
    if not text:
        raise ValueError(
            f"{setting} is not set: which label{where} is the vessel cannot be read from "
            f"ilastik's output. Fix: set {setting} to {choices}."
        )
    if names:
        if text in names:
            return list(names).index(text) + 1
        folded = [name.casefold() for name in names]
        if folded.count(text.casefold()) == 1:
            return folded.index(text.casefold()) + 1
    try:
        number = int(text)
    except ValueError:
        number = None
    if number is not None and number >= 1 and (not names or number <= len(names)):
        return number
    raise ValueError(
        f"{setting}={text!r} is not a label{where}. Fix: set it to {choices}."
    )


def file_sha256(path: str | Path) -> str:
    """The SHA-256 of *path*'s bytes, remembered while its size and
    modification time stay the same."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    known = _SHA256_MEMO.get(key)
    if known is not None:
        return known
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_BYTES), b""):
            digest.update(chunk)
    _SHA256_MEMO[key] = digest.hexdigest()
    return _SHA256_MEMO[key]


#: What the main image's segmentation adds to the raw image's stem.
SEGMENTED_STEM_SUFFIX = "_segmented"


def ilastik_segmented_stem(unsegmented_image_path: str | Path) -> str:
    """``{raw image stem}_segmented``: the stem of ilastik's segmentation of
    *unsegmented_image_path*, and so of every output of a run on it."""
    return f"{Path(unsegmented_image_path).stem}{SEGMENTED_STEM_SUFFIX}"


def ilastik_segmented_path(
    output_dir: str | Path, unsegmented_image_path: str | Path, suffix: str
) -> Path:
    """Where the main image's segmentation is asked for:
    ``{output_dir}/{raw image stem}_segmented{suffix}`` (ilastik may give a
    ``.tif`` back as ``.tiff``; the stem stays)."""
    return Path(output_dir) / f"{ilastik_segmented_stem(unsegmented_image_path)}{suffix}"


#: The folder, beside a vessel mask's ilastik output, the channels ilastik
#: reads are written to while it runs.
CHANNEL_INPUT_FOLDER = "channels"


def channel_input_path(output_path: str | Path) -> Path:
    """Where the chosen channels ilastik segments into *output_path* are
    written first: ``{output folder}/channels/{output stem}_input.tif``. Not
    the output's own folder, where ilastik writes its export, named after its
    input, before it is moved to *output_path*."""
    output_path = Path(output_path)
    return output_path.parent / CHANNEL_INPUT_FOLDER / f"{output_path.stem}_input.tif"


def write_ilastik_channels(
    image_path: str | Path, channels: Sequence[int], output_path: str | Path
) -> Path:
    """Write *channels* (counting from 0, as ``raw_image_channel``: C1 is 0)
    of the multi-channel TIFF at *image_path* to *output_path*, kept separate
    and in the order given, for ilastik to segment: the file's own axes, its
    channel axis holding these alone (dropped for one channel), the same kind
    of file (an ImageJ hyperstack stays one, anything else is written as
    OME-TIFF), its voxel size, and its voxel values untouched. Copied a plane
    at a time where the file holds a page per plane, so a large stack is never
    all in memory. Raises ``ValueError`` for a file with no channel axis, a
    channel it lacks, or a channel named twice."""
    import tifffile

    from haemolynx.io.channels import tiff_channel_axis
    from haemolynx.io.load import _voxel_size_xyz_from_tiff

    image_path, output_path = Path(image_path), Path(output_path)
    chosen = [int(channel) for channel in channels]
    if not chosen:
        raise ValueError("No channels chosen: leave the channels unset to give ilastik the whole file.")
    twice = sorted({channel for channel in chosen if chosen.count(channel) > 1})
    if twice:
        raise ValueError(f"Channel(s) {twice} are chosen more than once.")
    with tifffile.TiffFile(str(image_path)) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        shape = tuple(int(size) for size in series.shape)
        channel_axis = tiff_channel_axis(axes, shape, tif.imagej_metadata)
        if channel_axis is None:
            raise ValueError(
                f"{image_path.name} has no channels to choose from (axes {axes}): leave the "
                "channels unset, and ilastik reads the whole file."
            )
        count = shape[channel_axis]
        missing = [channel for channel in chosen if not 0 <= channel < count]
        if missing:
            raise ValueError(
                f"{image_path.name} has {count} channels (0-{count - 1}, C1-C{count}); "
                f"channel(s) {missing} do not exist."
            )
        x, y, z, _status = _voxel_size_xyz_from_tiff(tif)
        kept = list(shape)
        kept[channel_axis] = len(chosen)
        if len(chosen) == 1:
            new_axes = axes[:channel_axis] + axes[channel_axis + 1:]
            new_shape = tuple(kept[:channel_axis] + kept[channel_axis + 1:])
        else:
            new_axes, new_shape = axes, tuple(kept)
        leading = shape[:-2]
        if axes.endswith("YX") and channel_axis < len(leading) and len(series.pages) == int(
            np.prod(leading)
        ):
            pages = list(series.pages)

            def planes():
                for index in np.ndindex(*kept[:-2]):
                    source = list(index)
                    source[channel_axis] = chosen[index[channel_axis]]
                    yield pages[int(np.ravel_multi_index(source, leading))].asarray()

            data = planes()
        else:
            data = np.take(series.asarray(), chosen, axis=channel_axis).reshape(new_shape)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = dict(shape=new_shape, dtype=series.dtype, resolution=(1.0 / x, 1.0 / y))
        if tif.is_imagej:
            metadata = {"axes": new_axes, "spacing": z, "unit": "um"}
            if len(chosen) > 1:
                metadata["mode"] = "composite"
            tifffile.imwrite(output_path, data, imagej=True, metadata=metadata, **written)
        else:
            metadata = {
                "axes": new_axes, "PhysicalSizeX": x, "PhysicalSizeY": y, "PhysicalSizeZ": z,
            }
            tifffile.imwrite(
                output_path, data, ome=True, photometric="minisblack", metadata=metadata, **written
            )
    return output_path


def sidecar_path(output_path: str | Path) -> Path:
    """Where the record of how *output_path* was made is kept."""
    output_path = Path(output_path)
    return output_path.with_name(output_path.name + SIDECAR_SUFFIX)


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def ilastik_provenance(
    input_image_path: str | Path,
    classifier_path: str | Path,
    vessel_label: int,
    *,
    label_names: Sequence[str] | None = None,
    channels: Sequence[int] | None = None,
) -> dict:
    """What an output made from *input_image_path* by *classifier_path*,
    binarised at *vessel_label*, depends on: the project by its bytes, the
    input by path, size and modification time (hashing a raw volume of many
    GB on every run would cost more than the segmentation it saves), and the
    input's *channels* ilastik read, when it read only those."""
    classifier_path = Path(classifier_path)
    record = {
        "version": PROVENANCE_VERSION,
        "export_source": EXPORT_SOURCE,
        "project_sha256": file_sha256(classifier_path),
        "input": _file_identity(Path(input_image_path)),
        "vessel_label": int(vessel_label),
        "project": str(classifier_path.resolve()),
    }
    if channels:
        record["input"]["channels"] = [int(channel) for channel in channels]
    if label_names and 1 <= vessel_label <= len(label_names):
        record["vessel_label_name"] = label_names[vessel_label - 1]
    return record


def ilastik_output_is_current(output_path: str | Path, provenance: dict) -> tuple[bool, str]:
    """Whether the output at *output_path* was made as *provenance* says this
    run would make it, and why not: its sidecar must record the same project,
    input, export and vessel label, and the output must be the file the
    sidecar was written for (unchanged in size and time since)."""
    output_path = Path(output_path)
    if not output_path.is_file():
        return False, "there is no output yet"
    sidecar = sidecar_path(output_path)
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, (
            f"it has no {sidecar.name} saying how it was made (it was made before "
            "HaemoLynx recorded that, or by hand)"
        )
    except (OSError, ValueError):
        return False, f"its record {sidecar.name} cannot be read"
    if not isinstance(recorded, dict):
        return False, f"its record {sidecar.name} cannot be read"
    reasons = {
        "version": "it was recorded by another version of HaemoLynx",
        "export_source": "it is another kind of ilastik export",
        "project_sha256": "the ilastik project is not the one it was made with",
        "input": "the input image, or the channels read of it, has changed",
        "vessel_label": "it was binarised at another vessel label",
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


def _write_sidecar(output_path: Path, provenance: dict, *, vessel_voxels: int, voxels: int) -> None:
    stat = output_path.stat()
    record = dict(provenance)
    record["output"] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "vessel_voxels": int(vessel_voxels),
        "voxels": int(voxels),
    }
    sidecar_path(output_path).write_text(json.dumps(record, indent=2), encoding="utf-8")


def _binarise_volume(volume, label: int, *, description: str) -> tuple[int, int]:
    """Rewrite *volume* (writable, indexed along axis 0) as ``volume == label``
    in its own dtype, a slab at a time. Counted first, so a volume holding no
    voxel of *label* raises with it as ilastik wrote it."""
    from haemolynx.preprocessing.memmap_support import slab_step

    shape = tuple(volume.shape)
    step = slab_step(shape) if shape else 1
    tally: dict[int, int] = {}
    for start in range(0, shape[0] if shape else 1, step):
        slab = np.asarray(volume[start:start + step] if shape else volume[()])
        values, counts = np.unique(slab, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            tally[value] = tally.get(value, 0) + count
    vessel = tally.get(label, 0)
    voxels = sum(tally.values())
    if vessel == 0:
        held = ", ".join(str(value) for value in sorted(tally)) or "nothing"
        raise ValueError(
            f"{description} holds no voxel of the vessel label {label} (it holds {held}): "
            "the classifier found no vessel, or the vessel label setting names the wrong label."
        )
    for start in range(0, shape[0] if shape else 1, step):
        index = slice(start, start + step) if shape else ()
        slab = np.asarray(volume[index])
        volume[index] = (slab == label).astype(slab.dtype)
    return vessel, voxels


def binarise_ilastik_output(path: str | Path, label: int) -> tuple[int, int]:
    """Rewrite ilastik's Simple Segmentation export at *path* in place as a
    0/1 mask: 1 where it holds *label*, 0 elsewhere, in the same file, shape,
    dtype and metadata. Returns ``(vessel voxels, voxels)``.

    A volume holding no voxel of *label* raises ``ValueError`` and is left as
    ilastik wrote it: read as a mask it would be empty, or -- before the
    label was named -- solid vessel.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".h5":
        import h5py

        datasets: list = []
        with h5py.File(path, "r+") as handle:
            handle.visititems(
                lambda _name, item: datasets.append(item)
                if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.integer)
                else None
            )
            if not datasets:
                raise ValueError(f"ilastik's output {path} holds no integer dataset to binarise.")
            vessel = voxels = 0
            for dataset in datasets:
                counted = _binarise_volume(
                    dataset, label, description=f"ilastik's output {path} ({dataset.name})"
                )
                vessel, voxels = vessel + counted[0], voxels + counted[1]
        return vessel, voxels

    import tifffile

    try:
        volume = tifffile.memmap(path, mode="r+")
    except ValueError:
        # Not one contiguous, uncompressed block (ilastik's own OME-TIFF is):
        # read whole and written again with the same axes.
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            data = series.asarray()
            axes = series.axes
            ome = tif.is_ome
        counted = _binarise_volume(data, label, description=f"ilastik's output {path}")
        tifffile.imwrite(
            path, data, ome=ome, photometric="minisblack", metadata={"axes": axes}, software="ilastik"
        )
        return counted
    try:
        counted = _binarise_volume(volume, label, description=f"ilastik's output {path}")
        volume.flush()
    finally:
        del volume
    return counted


def _kill_process_tree(process: subprocess.Popen) -> None:
    """Stop *process* and anything it started.

    ``ilastik.exe`` (and a ``.bat``/shell wrapper round it) is a launcher that
    starts the real Python process; killing only the launcher leaves ilastik
    running and holding the output file open.
    """
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        else:
            import signal

            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - the OS refused
        logger.warning("ilastik (pid %s) did not exit after being killed.", process.pid)


def _read_lines(stream, lines: "queue.Queue[Optional[str]]") -> None:
    """Hand every line of *stream* to *lines*, then ``None`` at its end."""
    try:
        for line in iter(stream.readline, ""):
            lines.put(line.rstrip("\r\n"))
    finally:
        lines.put(None)


def run_ilastik_headless_segmentation(
    input_image_path: str | Path,
    classifier_path: str | Path,
    output_path: str | Path,
    ilastik_executable: str | Path = "ilastik.exe",
    timeout: float | None = None,
    *,
    vessel_label: object,
    label_setting: str = "vessel_label",
    reuse_existing: bool = False,
    poll: Optional[Callable[[], None]] = None,
    channels: Sequence[int] | None = None,
) -> Path:
    """Run ilastik headless segmentation for a single image, and leave its
    output as a 0/1 vessel mask.

    Parameters
    ----------
    input_image_path:
        Path to the image file to segment.
    classifier_path:
        Path to the ilastik project/classifier file (.ilp).
    output_path:
        Path to write the segmented output image.
    ilastik_executable:
        ilastik executable. Defaults to ``ilastik.exe`` and can also be a full path.
    timeout:
        Kill the subprocess and raise if it runs longer than this many
        seconds. ``None`` (this function's own default) waits forever; the
        pipeline itself always passes ``ilastik_timeout_seconds`` so a hung
        or misconfigured ilastik process cannot block a run indefinitely.
    vessel_label:
        Which of the project's labels is the vessel: a label name, or its
        number (1 for the first label made). Resolved against the names the
        ``.ilp`` holds (:func:`resolve_vessel_label`); the output is rewritten
        as 1 where ilastik gave that label and 0 elsewhere
        (:func:`binarise_ilastik_output`).
    label_setting:
        The setting *vessel_label* came from, named in its errors.
    reuse_existing:
        Return *output_path* without running ilastik when its sidecar
        (:func:`sidecar_path`) says it was made from this image, by a project
        with these bytes, at this vessel label, and it has not changed since
        (:func:`ilastik_output_is_current`). An output with no sidecar, or
        another, is segmented again, with a warning saying why.
    poll:
        Called about every :data:`POLL_INTERVAL_SECONDS` while ilastik runs.
        Whatever it raises stops the run: ilastik (and anything it started) is
        killed, and the exception propagates. This is how the napari panel's
        stop reaches a segmentation that can take hours.
    channels:
        The channels of a multi-channel TIFF ilastik reads, counting from 0,
        kept separate and in this order (:func:`write_ilastik_channels`, into
        :func:`channel_input_path` for as long as ilastik runs); None or empty
        for the whole file. The output's record names them, so other channels
        segment again.

    ilastik's own output is logged line by line as it arrives (``INFO``), so a
    long segmentation shows its progress in the run log rather than nothing
    until it ends.
    """
    input_image_path = Path(input_image_path)
    classifier_path = Path(classifier_path)
    output_path = Path(output_path)

    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier/project file not found: {classifier_path}")

    output_suffix = output_path.suffix.lower()
    if output_suffix not in OUTPUT_FORMATS:
        raise ValueError(
            "Unsupported ilastik output extension. "
            "Use one of: .tif, .tiff, .h5"
        )

    label_names = ilastik_label_names(classifier_path)
    label = resolve_vessel_label(
        vessel_label, label_names, setting=label_setting, project=classifier_path
    )
    provenance = ilastik_provenance(
        input_image_path, classifier_path, label, label_names=label_names, channels=channels
    )
    if reuse_existing:
        current, reason = ilastik_output_is_current(output_path, provenance)
        if current:
            logger.info(
                "Reusing ilastik output %s: made from %s by this %s (same SHA-256), vessel "
                "label %d. Delete it, or turn off ilastik_reuse_existing_output, to segment again.",
                output_path,
                input_image_path.name,
                classifier_path.name,
                label,
            )
            return output_path
        if output_path.exists():
            logger.warning("Segmenting %s again: %s.", output_path, reason)

    ilastik_input = input_image_path
    if channels:
        ilastik_input = write_ilastik_channels(
            input_image_path, channels, channel_input_path(output_path)
        )
        logger.info(
            "ilastik reads channel(s) %s (C%s) of %s, written for it to %s.",
            ", ".join(str(int(c)) for c in channels),
            ", C".join(str(int(c) + 1) for c in channels),
            input_image_path.name,
            ilastik_input,
        )
    try:
        return _segment(
            ilastik_input,
            classifier_path,
            output_path,
            ilastik_executable,
            timeout,
            poll=poll,
            label=label,
            provenance=provenance,
        )
    finally:
        if ilastik_input != input_image_path:
            _discard_channel_input(ilastik_input)


def _discard_channel_input(path: Path) -> None:
    """Delete a channel file written for ilastik, and its folder once empty:
    the output's record says which channels it was made from."""
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _segment(
    input_image_path: Path,
    classifier_path: Path,
    output_path: Path,
    ilastik_executable: str | Path,
    timeout: float | None,
    *,
    poll: Optional[Callable[[], None]],
    label: int,
    provenance: dict,
) -> Path:
    """Run ilastik on *input_image_path* into *output_path*, binarise it at
    *label* and write its record (*provenance*): what
    :func:`run_ilastik_headless_segmentation` does once nothing can be reused."""
    output_suffix = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # An old output left in place would be returned as this run's if ilastik
    # exited cleanly without writing one, and its record would vouch for it.
    output_path.unlink(missing_ok=True)
    sidecar_path(output_path).unlink(missing_ok=True)
    export_format, export_extension = OUTPUT_FORMATS[output_suffix]
    # The name ilastik writes to, moved to *output_path* afterwards: one an
    # earlier run left there would be moved in the same way.
    generated_path = output_path.parent / f"{input_image_path.stem}{export_extension}"
    generated_path.unlink(missing_ok=True)
    # ilastik expects output placeholders in the pattern string.
    output_pattern = str(output_path.parent / f"{{nickname}}{export_extension}")
    command = [
        str(ilastik_executable),
        "--headless",
        # Never save the project: the sidecar records its bytes.
        "--readonly",
        f"--project={classifier_path}",
        f"--export_source={EXPORT_SOURCE}",
        f"--output_format={export_format}",
        f"--output_filename_format={output_pattern}",
        str(input_image_path),
    ]
    popen_kwargs: dict = {}
    if os.name != "nt":
        # Its own process group, so a kill reaches whatever the launcher starts.
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find ilastik executable '{ilastik_executable}'. "
            "Set the ilastik_executable setting to the full ilastik path."
        ) from exc

    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    reader = threading.Thread(
        target=_read_lines, args=(process.stdout, lines), daemon=True
    )
    reader.start()
    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    started = time.monotonic()
    stream_open = True

    def drain(wait: float) -> None:
        nonlocal stream_open
        deadline = time.monotonic() + wait
        while stream_open:
            remaining = deadline - time.monotonic()
            try:
                if remaining > 0:
                    line = lines.get(timeout=remaining)
                else:
                    line = lines.get_nowait()
            except queue.Empty:
                return
            if line is None:
                stream_open = False
                return
            tail.append(line)
            if line.strip():
                logger.info("[ilastik] %s", line)

    try:
        while True:
            drain(POLL_INTERVAL_SECONDS)
            if process.poll() is not None and not stream_open:
                break
            if process.poll() is not None:
                # Exited; collect whatever it wrote last, then stop.
                drain(2.0)
                break
            if poll is not None:
                poll()
            if timeout is not None and time.monotonic() - started > float(timeout):
                _kill_process_tree(process)
                raise RuntimeError(
                    f"ilastik segmentation did not finish within {timeout} seconds "
                    "and was killed. Raise ilastik_timeout_seconds if this is a "
                    "genuinely large volume, or check the project file and input "
                    "for what is making ilastik hang.\n"
                    f"Command: {' '.join(command)}"
                )
    except BaseException:
        _kill_process_tree(process)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()

    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            "ilastik segmentation failed "
            f"(exit code {returncode}).\n"
            f"Command: {' '.join(command)}\n"
            "Last output:\n" + "\n".join(tail)
        )

    if generated_path.exists():
        if generated_path != output_path:
            generated_path.replace(output_path)
    if not output_path.exists():
        # ilastik can fail and still exit 0 (a feature scale too large for
        # the image), so what it said last is the explanation.
        raise RuntimeError(
            "ilastik finished, but no output file was found at "
            f"'{output_path}'. Expected generated path: '{generated_path}'.\n"
            "Last output:\n" + "\n".join(tail)
        )
    vessel_voxels, voxels = binarise_ilastik_output(output_path, label)
    _write_sidecar(output_path, provenance, vessel_voxels=vessel_voxels, voxels=voxels)
    name = provenance.get("vessel_label_name")
    logger.info(
        "ilastik output %s binarised at vessel label %d%s: %.1f%% of its voxels are vessel.",
        output_path,
        label,
        f" ({name!r})" if name else "",
        100.0 * vessel_voxels / max(voxels, 1),
    )
    return output_path
