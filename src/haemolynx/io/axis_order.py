"""Input axis-order handling and canonical (z, y, x) conventions.

HaemoLynx works internally in a canonical ``(z, y, x)`` array layout:

* array axis 0 is ``z`` — the stack axis, i.e. the axis that overlays and
  maximum-intensity projections are projected *through*,
* array axis 1 is ``y``,
* array axis 2 is ``x``.

Node ``pos`` attributes, edge ``voxels`` paths and every per-array-axis spacing
vector follow that same ``(z, y, x)`` order. Voxel sizes read from file
metadata, by contrast, are physically labelled and are therefore carried as
``(x, y, z)``; :func:`voxel_size_zyx_from_xyz` converts between the two. Mixing
the two orders silently mis-scales anisotropic data, so conversion happens once,
at the loader boundary.

Input files do not always arrive in ``(z, y, x)``. Loaders accept an
``axis_order`` string naming what each *array* axis means (``"xyz"`` for an
(x, y, z) volume, ``"yzx"`` for a (y, z, x) volume, and so on) and transpose the
volume to the canonical order on load. That is what makes "which axis is z" —
the axis projected through — user-selectable.

``image_axis_order="auto"`` reads each file in the order its own axis labels
give (:func:`stated_axis_order`: a TIFF's OME or ImageJ metadata, an HDF5
dataset's ``axistags``), and a file whose labels do not name z, y and x in the
segmented image's order, else ``"zyx"`` (:func:`axis_order_for_run`). Loaders
take ``"auto"`` too and resolve it against the file they open
(:func:`axis_order_for_file`); anything that transposes an array without a
file needs a permutation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

CANONICAL_AXIS_ORDER = "zyx"
AUTO_AXIS_ORDER = "auto"
_AUTO_PREFIX = f"{AUTO_AXIS_ORDER}:"
VALID_AXIS_ORDERS = (
    "zyx",
    "zxy",
    "yzx",
    "yxz",
    "xzy",
    "xyz",
)


def normalize_axis_order(
    axis_order: str, *, label: str = "axis_order", allow_auto: bool = False
) -> str:
    """Validate an axis-order string and return it lowercased.

    The string names the meaning of each array axis in order, so ``"zyx"``
    describes a volume indexed ``volume[z, y, x]``. *allow_auto* accepts an
    auto order as well (``"auto"``, or :func:`auto_axis_order`'s
    ``"auto:<order>"``), for a caller that resolves it against a file.
    """
    if not isinstance(axis_order, str):
        raise ValueError(
            f"{label} must be a string permutation of 'xyz', got {axis_order!r}."
        )
    normalized = axis_order.strip().lower()
    if is_auto_axis_order(normalized):
        if not allow_auto:
            raise ValueError(
                f"{label}={axis_order!r} names no order until it is resolved against "
                "a file (axis_order_for_file); a permutation of 'xyz' is needed here."
            )
        unlabelled = normalized[len(_AUTO_PREFIX):] or CANONICAL_AXIS_ORDER
        return auto_axis_order(normalize_axis_order(unlabelled, label=label))
    if sorted(normalized) != ["x", "y", "z"]:
        auto = " or 'auto'" if allow_auto else ""
        raise ValueError(
            f"{label} must be a permutation of 'xyz' (one of {', '.join(VALID_AXIS_ORDERS)})"
            f"{auto}, got {axis_order!r}."
        )
    return normalized


def is_auto_axis_order(axis_order: Any) -> bool:
    """Whether *axis_order* is an auto order: ``"auto"`` or ``"auto:<order>"``."""
    if not isinstance(axis_order, str):
        return False
    normalized = axis_order.strip().lower()
    return normalized == AUTO_AXIS_ORDER or normalized.startswith(_AUTO_PREFIX)


def auto_axis_order(unlabelled: str = CANONICAL_AXIS_ORDER) -> str:
    """The auto order that reads a file whose labels do not name z, y and x
    in *unlabelled*: ``"auto"`` while that is ``"zyx"``, else
    ``"auto:<unlabelled>"``. A run under ``image_axis_order="auto"`` hands
    its loaders this with the segmented image's order
    (:func:`axis_order_for_run`)."""
    unlabelled = normalize_axis_order(unlabelled, label="unlabelled")
    if unlabelled == CANONICAL_AXIS_ORDER:
        return AUTO_AXIS_ORDER
    return f"{_AUTO_PREFIX}{unlabelled}"


def axis_order_from_labels(labels: Iterable[str]) -> str | None:
    """The order of z, y and x among a file's axis *labels*, one per array
    axis (``"ZCYX"``, ``["z", "c", "y", "x"]``), its other axes -- channel,
    time, samples -- left out; None unless z, y and x are each named once.
    tifffile names an axis it knows nothing about ``Q`` or ``I``, so a stack
    written without metadata (``"QYX"``) gives None."""
    spatial = [str(label).strip().lower() for label in labels]
    spatial = [label for label in spatial if label in {"x", "y", "z"}]
    if sorted(spatial) != ["x", "y", "z"]:
        return None
    return "".join(spatial)


def resolve_axis_order(axis_order: str, stated: str | None) -> str:
    """The permutation a file is read in: *axis_order* when it is one; for
    an auto order, the order the file's labels state (*stated*), else the
    auto order's unlabelled one."""
    normalized = normalize_axis_order(axis_order, allow_auto=True)
    if not is_auto_axis_order(normalized):
        return normalized
    if stated is not None:
        return normalize_axis_order(stated, label="stated axis order")
    return normalized[len(_AUTO_PREFIX):] or CANONICAL_AXIS_ORDER


_STATED_AXIS_ORDERS: dict[tuple, str | None] = {}
_STATED_AXIS_ORDERS_KEPT = 64
_LOGGED_RESOLUTIONS: set[tuple] = set()


def _h5_axis_labels(path: Path, dataset_name: str | None) -> list[str] | None:
    try:
        import h5py
    except ImportError:
        return None
    from .load import _h5_axis_keys, _select_h5_dataset

    with h5py.File(path, "r") as f:
        return _h5_axis_keys(f[_select_h5_dataset(f, path, dataset_name)])


def stated_axis_order(path: Any, dataset_name: str | None = None) -> str | None:
    """The order a file's own axis labels give its z, y and x axes; None when
    they do not name all three, or *path* is not a readable TIFF or HDF5 file.

    A TIFF's labels are tifffile's for its first series: from the OME header,
    ImageJ's hyperstack metadata or a tifffile writer's ``axes``. Every TIFF
    page is a Y-by-X plane, so a TIFF states z first unless its writer named
    the axes otherwise, and a resliced stack still labelled ``ZYX`` is not
    caught. tifffile writes an ImageJ stack given no axes as channels
    (``CYX``), which names no z. An HDF5 dataset's (*dataset_name*, or the one the loader picks)
    are the ``axistags`` ilastik and vigra write, else h5py's dimension
    labels, in any order. Read from the header alone, and kept per file
    until the file changes.
    """
    if path is None or path == "":
        return None
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
    except (OSError, TypeError, ValueError):
        return None
    key = (str(resolved), stat.st_mtime_ns, stat.st_size, dataset_name)
    if key in _STATED_AXIS_ORDERS:
        return _STATED_AXIS_ORDERS[key]
    labels = None
    suffix = resolved.suffix.lower()
    try:
        if suffix in {".tif", ".tiff"}:
            with tifffile.TiffFile(resolved) as tif:
                labels = tif.series[0].axes
        elif suffix in {".h5", ".hdf5"}:
            labels = _h5_axis_labels(resolved, dataset_name)
    except Exception:  # noqa: BLE001 - an unreadable file states nothing; its loader says why
        logger.debug("Could not read the axis labels of %s", resolved, exc_info=True)
    stated = None if labels is None else axis_order_from_labels(labels)
    if len(_STATED_AXIS_ORDERS) >= _STATED_AXIS_ORDERS_KEPT:
        _STATED_AXIS_ORDERS.pop(next(iter(_STATED_AXIS_ORDERS)))
    _STATED_AXIS_ORDERS[key] = stated
    return stated


def axis_order_for_file(
    axis_order: str, path: Any, *, dataset_name: str | None = None
) -> str:
    """The permutation *path* is read in under *axis_order*: the order itself,
    or for an auto order, the file's own labels' (:func:`stated_axis_order`),
    else the auto order's unlabelled one. Every loader calls this before it
    transposes; under auto, which order each file got, and why, is logged
    once."""
    normalized = normalize_axis_order(axis_order, allow_auto=True)
    if not is_auto_axis_order(normalized):
        return normalized
    stated = stated_axis_order(path, dataset_name)
    resolved = resolve_axis_order(normalized, stated)
    logged = (str(path), resolved, stated is not None)
    if path not in (None, "") and logged not in _LOGGED_RESOLUTIONS and Path(path).is_file():
        _LOGGED_RESOLUTIONS.add(logged)
        if stated is not None:
            logger.info(
                "image_axis_order=auto: reading %s as '%s', as its axis labels say.",
                Path(path).name, resolved,
            )
        else:
            logger.info(
                "image_axis_order=auto: %s does not name its z, y and x axes; reading it as '%s'.",
                Path(path).name, resolved,
            )
    return resolved


def axis_order_for_run(settings: Mapping[str, Any]) -> str:
    """The *axis_order* a run hands its loaders: ``image_axis_order`` when it
    is a permutation; under ``"auto"``, the auto order whose files without
    labels are read in the segmented image's (``input_path``) order -- its
    own labels', else ``"zyx"``. Each loader resolves it against its own file
    (:func:`axis_order_for_file`)."""
    value = settings.get("image_axis_order") or CANONICAL_AXIS_ORDER
    normalized = normalize_axis_order(value, label="image_axis_order", allow_auto=True)
    if not is_auto_axis_order(normalized):
        return normalized
    return auto_axis_order(axis_order_for_file(normalized, settings.get("input_path")))


def axis_order_transpose(axis_order: str) -> tuple[int, int, int]:
    """Return the ``np.transpose`` axes that map *axis_order* to ``(z, y, x)``."""
    normalized = normalize_axis_order(axis_order)
    return tuple(normalized.index(axis) for axis in CANONICAL_AXIS_ORDER)


def apply_axis_order(
    volume: np.ndarray,
    axis_order: str = CANONICAL_AXIS_ORDER,
    *,
    use_memmap: bool = False,
    memmap_directory=None,
) -> np.ndarray:
    """Transpose *volume* from *axis_order* into the canonical ``(z, y, x)`` order.

    ``axis_order="zyx"`` is a no-op. Other orders return a contiguous copy so
    downstream skeletonization and distance transforms are not slowed by a
    non-contiguous view.

    Uses ``asanyarray``, not ``asarray``, for the no-op case specifically so
    it is a genuine no-op even for an ``ndarray`` subclass -- a
    ``numpy.memmap`` loaded with ``use_memmap_loading`` would otherwise come
    back demoted to a plain in-RAM-looking ``ndarray`` (same underlying
    disk-backed buffer, since it is still a view rather than a copy, but the
    wrong type for anything downstream that checks ``isinstance(x,
    np.memmap)`` to decide whether it owns a backing file to release).

    *use_memmap*, when True, writes that contiguous copy into a new memmap in
    *memmap_directory* instead of ``np.ascontiguousarray``, which is a
    whole-volume plain-RAM copy no matter how *volume* itself is stored. It
    is copied one 3D block at a time, not one output slice at a time: an
    output slice gathers from across the whole source file, so slice by
    slice re-reads the source from disk once per slice as soon as it no
    longer fits the page cache, where a block reads only its own region.
    """
    normalized = normalize_axis_order(axis_order)
    arr = np.asanyarray(volume)
    if normalized == CANONICAL_AXIS_ORDER:
        return arr
    if arr.ndim != 3:
        raise ValueError(
            f"axis_order={axis_order!r} requires a 3D volume, got shape {arr.shape}."
        )
    transposed = np.transpose(arr, axis_order_transpose(normalized))
    if not use_memmap:
        return np.ascontiguousarray(transposed)
    from haemolynx.preprocessing.memmap_support import new_memmap_array

    from haemolynx.preprocessing.memmap_support import iter_blocks

    result = new_memmap_array(transposed.shape, transposed.dtype, directory=memmap_directory)
    for _padded, _inner, core in iter_blocks(transposed.shape, halo=0):
        result[core] = transposed[core]
    return result


def voxel_size_zyx_from_xyz(
    voxel_size_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert a physical ``(x, y, z)`` voxel size to canonical array order ``(z, y, x)``.

    Use this wherever a voxel size read from image metadata is handed to code
    that scales array indices — graph construction, distance transforms, FWHM
    sampling. Passing ``(x, y, z)`` straight through mis-scales anisotropic
    volumes by swapping the z and x spacings.
    """
    arr = np.asarray(voxel_size_xyz, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(
            f"voxel_size_xyz must have length 3, got {voxel_size_xyz}."
        )
    return (float(arr[2]), float(arr[1]), float(arr[0]))


def voxel_size_xyz_from_zyx(
    voxel_size_zyx: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert a canonical array-order ``(z, y, x)`` voxel size back to ``(x, y, z)``."""
    arr = np.asarray(voxel_size_zyx, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(
            f"voxel_size_zyx must have length 3, got {voxel_size_zyx}."
        )
    return (float(arr[2]), float(arr[1]), float(arr[0]))


def voxel_size_xyz_from_file_axes(
    file_axis_spacing: Sequence[float], axis_order: str = CANONICAL_AXIS_ORDER
) -> tuple[float, float, float]:
    """Physical ``(x, y, z)`` voxel size from the spacing along each of a
    file's own array axes, in the file's order, given *axis_order* -- what
    each of those axes is.

    A TIFF's resolution tags and an HDF5 ``element_size_um`` describe the
    file's axes (pages, height, width), not physical ones: in a stack saved
    with ``image_axis_order="xyz"`` the page spacing is the x spacing. Read
    as if the file were ``zyx``, the z and x spacings swap for the whole run.
    For the default ``"zyx"`` this is :func:`voxel_size_xyz_from_zyx`.
    """
    normalized = normalize_axis_order(axis_order)
    spacing = [float(value) for value in np.asarray(file_axis_spacing, dtype=float).ravel()]
    if len(spacing) != 3:
        raise ValueError(f"file_axis_spacing must have length 3, got {file_axis_spacing}.")
    by_axis = dict(zip(normalized, spacing))
    return (by_axis["x"], by_axis["y"], by_axis["z"])


def file_axis_spacing_from_xyz(
    voxel_size_xyz: Sequence[float], axis_order: str = CANONICAL_AXIS_ORDER
) -> tuple[float, float, float]:
    """The inverse of :func:`voxel_size_xyz_from_file_axes`: the spacing along
    each of a file's own array axes, in the file's order."""
    normalized = normalize_axis_order(axis_order)
    values = [float(value) for value in np.asarray(voxel_size_xyz, dtype=float).ravel()]
    if len(values) != 3:
        raise ValueError(f"voxel_size_xyz must have length 3, got {voxel_size_xyz}.")
    by_axis = dict(zip("xyz", values))
    return tuple(by_axis[axis] for axis in normalized)
